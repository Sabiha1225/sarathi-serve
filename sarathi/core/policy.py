from typing import List, Optional

from sarathi.core.datatypes.sequence import Sequence

import math
import os

EPS = 1e-6
DEFAULT_BLOCK_SIZE = 16
DEFAULT_DEADLINE_SLACK_SEC = 30.0

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _prompt_remaining(seq: Sequence) -> int:
    return max(seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(), 0)


def _decode_remaining(seq: Sequence) -> int:
    max_tokens = getattr(seq.sampling_params, "max_tokens", 0)
    return max(max_tokens - seq.get_output_len(), 0)


def _total_work(seq: Sequence) -> int:
    # static: original prompt + max_tokens, does not shrink as the request executes
    return seq.get_prompt_len() + getattr(seq.sampling_params, "max_tokens", 0)


def _remaining_work(seq: Sequence) -> int:
    # dynamic: shrinks as prefill/decode progress — this is what makes SRPT
    # different from SJF (SJF never reorders mid-flight, SRPT can)
    return _prompt_remaining(seq) + _decode_remaining(seq)


def _waiting_time(now: float, seq: Sequence) -> float:
    return max(now - seq.arrival_time, 0.0)


def _get_user_priority(seq: Sequence) -> float:
    """Requires sampling_params.priority to be set (see prerequisite above).
    Falls back to 0.0 for every request if not wired — which makes Priority
    degenerate to "preserve prior order," not a real priority policy."""
    sampling_params = getattr(seq, "sampling_params", None)
    return float(getattr(sampling_params, "priority", 0.0) or 0.0)


def _get_deadline(now: float, seq: Sequence) -> float:
    """Requires sampling_params.ttft_slo_ms/tpot_slo_ms to be set. Falls back
    to a single global env-var-based slack for every request — which makes
    EDF degenerate to FCFS (everyone has the same deadline offset from their
    own arrival time, so deadline order == arrival order)."""
    sampling_params = getattr(seq, "sampling_params", None)
    ttft_slo_ms = getattr(sampling_params, "ttft_slo_ms", None)
    if ttft_slo_ms is not None:
        return seq.arrival_time + ttft_slo_ms / 1000.0

    slo_sec = float(
        os.getenv("SARATHI_SCHEDULER_DEFAULT_DEADLINE_SEC", str(DEFAULT_DEADLINE_SLACK_SEC))
    )
    return seq.arrival_time + slo_sec


def _laxity(now: float, seq: Sequence) -> float:
    """Time remaining before this request breaches its own SLO, tracking
    whichever deadline currently applies (TTFT while in prefill, TPOT-derived
    inter-token deadline once decoding)."""
    sampling_params = getattr(seq, "sampling_params", None)

    if not seq.prompt_processing_finished:
        ttft_slo_ms = getattr(sampling_params, "ttft_slo_ms", None)
        ttft_slo_s = (ttft_slo_ms / 1000.0) if ttft_slo_ms is not None else DEFAULT_DEADLINE_SLACK_SEC
        return ttft_slo_s - _waiting_time(now, seq)

    tpot_slo_ms = getattr(sampling_params, "tpot_slo_ms", None)
    tpot_slo_s = (tpot_slo_ms / 1000.0) if tpot_slo_ms is not None else 0.1
    # last_token_time = getattr(seq, "_last_token_generated_at_public", None) or seq.arrival_time
    last_token_time = seq.state.last_token_generated_at or seq.arrival_time
    return tpot_slo_s - max(now - last_token_time, 0.0)


def _block_size(seq: Sequence, scheduler=None) -> int:
    if scheduler is not None:
        return scheduler.cache_config.block_size
    return DEFAULT_BLOCK_SIZE

class Policy:

    def get_priority(
        self,
        now: float,
        seq: Sequence,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seqs: List[Sequence],
    ) -> List[Sequence]:
        return sorted(
            seqs,
            key=lambda seq: self.get_priority(now, seq),
            reverse=True,
        )

    def sort_admission(
        self, now: float, seqs: List[Sequence], scheduler=None
    ) -> List[Sequence]:
        """Separate hook for ordering the WAITING (admission) queue, in case
        a policy wants a different rule for "who gets admitted next" vs "who
        gets to keep their slot under memory pressure." Defaults to the same
        formula as sort_by_priority; override per-policy if you want them to
        diverge."""
        return self.sort_by_priority(now, seqs, scheduler)


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq: Sequence,
    ) -> float:
        return now - seq.arrival_time


class SJF(Policy):
    """Static shortest-job-first: total original work, oracle-valid for your
    traces (ignore_eos=True + max_tokens=num_decode_tokens => exact ground
    truth). Never reorders mid-flight."""

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return -float(_total_work(seq))


class LJF(Policy):
    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return float(_total_work(seq))


class SRPT(Policy):
    """Shortest-remaining-processing-time: dynamic, reorders as requests
    progress. This is what my earlier "SJF" actually computed — kept here
    under its correct name."""

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return -float(_remaining_work(seq))


class PromptLenSJF(Policy):
    """Realistic (non-oracle) SJF proxy matching vLLM RFC #29406: prompt
    length only (known at admission), time-weighted so it doesn't starve
    long requests."""

    def __init__(self, aging_weight: float = 0.01):
        self.aging_weight = aging_weight

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return -seq.get_prompt_len() + self.aging_weight * _waiting_time(now, seq)


class Priority(Policy):
    """Pure tenant/user priority tier. Requires sampling_params.priority to
    be populated — see prerequisite note."""

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return _get_user_priority(seq)


class EDF(Policy):
    """Earliest-deadline-first. Requires sampling_params.ttft_slo_ms /
    tpot_slo_ms to be populated, otherwise degenerates to FCFS — see
    _get_deadline's docstring."""

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return -float(_get_deadline(now, seq))


class LeastLaxityFirst(Policy):
    """Real-time-scheduling laxity: prioritizes whoever is closest to
    breaching their own SLO right now, tracking TTFT while in prefill and
    TPOT once decoding. More responsive than static EDF since it re-evaluates
    the *current* applicable deadline each call instead of a fixed one."""

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return -_laxity(now, seq)


# ---------------------------------------------------------------------------
# Aging / blended (published precedents)
# ---------------------------------------------------------------------------
class AgingSJF(Policy):
    """Weighted-Shortest-Job-First (WSJF) ratio form: wait_time / work.
    Different dynamics from additive aging — blows up as work -> 0."""

    def __init__(self, aging_weight: float = 1.0):
        self.aging_weight = aging_weight

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        work = max(_total_work(seq), 1)
        return (_waiting_time(now, seq) * self.aging_weight) / work


class AgingSRPT(Policy):
    def __init__(self, aging_weight: float = 1.0):
        self.aging_weight = aging_weight

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        work = max(_remaining_work(seq), 1)
        return (_waiting_time(now, seq) * self.aging_weight) / work


class SkipJoinMLFQ(Policy):
    """FastServe-style: priority driven by service already received, with
    aging. Doesn't need to know true length at all."""

    def __init__(self, aging_factor: float = 0.001):
        self.aging_factor = aging_factor

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        tokens_served = seq.get_num_prompt_tokens_processed() + seq.get_output_len()
        return -tokens_served + self.aging_factor * _waiting_time(now, seq)


class PrefillAgingFairness(Policy):
    """Matches arXiv:2606.09061 (chunked-prefill-native aging): wait_time -
    remaining_prefill_work. Input-side only, blind to decode length — your
    strongest published baseline to beat, not a strawman."""

    def __init__(self, wait_weight: float = 1.0, prefill_weight: float = 1.0):
        self.wait_weight = wait_weight
        self.prefill_weight = prefill_weight

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        return self.wait_weight * _waiting_time(now, seq) - self.prefill_weight * _prompt_remaining(seq)


class VTCFairness(Policy):
    """OSDI'24 VTC-style: prioritize least-service-received client so far.
    Falls back to per-request fairness (single-request "clients") if seq has
    no client_id attribute."""

    def __init__(self, input_token_cost: float = 1.0, output_token_cost: float = 1.0):
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost
        self._service_received = {}

    def _cost(self, seq: Sequence) -> float:
        return (
            self.input_token_cost * seq.get_num_prompt_tokens_processed()
            + self.output_token_cost * seq.get_output_len()
        )

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        client_id = getattr(seq, "client_id", seq.seq_id)
        service_so_far = self._service_received.get(client_id, 0.0) + self._cost(seq)
        self._service_received[client_id] = service_so_far
        return -service_so_far


# ---------------------------------------------------------------------------
# Phase / resource-pressure controls
# ---------------------------------------------------------------------------
class DecodeFirst(Policy):
    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        bonus = math.inf if seq.prompt_processing_finished else 0.0
        return bonus + _waiting_time(now, seq)


class PrefillFirst(Policy):
    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        bonus = math.inf if not seq.prompt_processing_finished else 0.0
        return bonus + _waiting_time(now, seq)


class MemoryAwareSJF(Policy):
    """SJF penalized by KV-block footprint, so memory-heavy requests are
    deprioritized even if their token count alone looks short."""

    def __init__(self, block_penalty: float = 1.0):
        self.block_penalty = block_penalty

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        blocks = len(seq.logical_token_blocks)
        block_size = _block_size(seq, scheduler)
        work = max(_total_work(seq), 1)
        return -float(work + blocks * block_size * self.block_penalty)


# ---------------------------------------------------------------------------
# Static multi-factor blend (published-precedent category: fixed weights,
# e.g. vLLM RFC #29406, arXiv:2606.09061, Ascendra)
# ---------------------------------------------------------------------------
class HybridAdaptive(Policy):
    """NOTE: despite the name, this is a STATIC weighted blend — weights are
    fixed for the whole run, not a function of load. Kept as a baseline
    representing "combine everything with fixed weights." Weights are
    constructor kwargs (not env vars) so they're captured in your run's
    config.yml snapshot for reproducibility."""

    def __init__(
        self,
        aging_weight: float = 1.0,
        deadline_weight: float = 1.0,
        priority_weight: float = 1000.0,
        work_weight: float = 1.0,
        decode_bonus: float = 100.0,
    ):
        self.aging_weight = aging_weight
        self.deadline_weight = deadline_weight
        self.priority_weight = priority_weight
        self.work_weight = work_weight
        self.decode_bonus = decode_bonus

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        remaining = max(_remaining_work(seq), 1)
        time_to_deadline = max(_get_deadline(now, seq) - now, EPS)

        aging_score = self.aging_weight * (_waiting_time(now, seq) / remaining)
        deadline_score = self.deadline_weight * (1.0 / time_to_deadline)
        priority_score = self.priority_weight * _get_user_priority(seq)
        work_score = -self.work_weight * remaining
        decode_score = self.decode_bonus if seq.prompt_processing_finished else 0.0

        return aging_score + deadline_score + priority_score + work_score + decode_score


# ---------------------------------------------------------------------------
# NEW — the one policy in this file that is actually load-adaptive (weights
# are a function of live queue occupancy, not fixed). Not found in FastServe,
# VTC, S3, Response-Length-Perception, Learning-to-Rank, arXiv:2606.09061,
# Ascendra, or vLLM RFC #29406 — all of those use a static formula.
# ---------------------------------------------------------------------------
class AdaptiveInputOutputAging(Policy):
    """priority = w_age(load)*wait - w_len(load)*(remaining_prefill +
    predicted_remaining_decode) + w_slo*urgency

    At low load: collapses toward pure aging (~FCFS) since reordering only
    costs fairness with no throughput benefit under no contention.
    At high load: shifts toward remaining-work terms (~SRPT) to protect
    throughput/short jobs from head-of-line blocking.
    SLO urgency term is independent of load — an imminent SLO breach should
    override the blend regardless of current occupancy.
    """

    def __init__(
        self,
        length_predictor=None,
        low_load_threshold: float = 0.3,
        high_load_threshold: float = 0.8,
        max_aging_weight: float = 1.0,
        max_length_weight: float = 1.0,
        slo_weight: float = 0.0,  # set > 0 to activate the SLO term
    ):
        # default = oracle remaining decode length, valid for your synthetic
        # traces (ignore_eos=True => max_tokens is exact ground truth)
        self.length_predictor = length_predictor or (
            lambda seq: seq.sampling_params.max_tokens - seq.get_output_len()
        )
        self.low_load_threshold = low_load_threshold
        self.high_load_threshold = high_load_threshold
        self.max_aging_weight = max_aging_weight
        self.max_length_weight = max_length_weight
        self.slo_weight = slo_weight

    def _system_load(self, scheduler) -> float:
        if scheduler is None:
            return 0.0
        max_seqs = max(1, scheduler.scheduler_config.max_num_seqs)
        occupancy = (len(scheduler.running) + len(scheduler.waiting)) / max_seqs
        return min(1.0, occupancy)

    def _blend_weights(self, load: float):
        if load <= self.low_load_threshold:
            t = 0.0
        elif load >= self.high_load_threshold:
            t = 1.0
        else:
            t = (load - self.low_load_threshold) / (self.high_load_threshold - self.low_load_threshold)
        return self.max_aging_weight * (1.0 - t), self.max_length_weight * t

    def get_priority(self, now: float, seq: Sequence, scheduler=None) -> float:
        load = self._system_load(scheduler)
        aging_weight, length_weight = self._blend_weights(load)

        remaining_prefill = _prompt_remaining(seq)
        predicted_remaining_decode = max(0, self.length_predictor(seq))
        urgency = -_laxity(now, seq) if self.slo_weight else 0.0

        return (
            aging_weight * _waiting_time(now, seq)
            - length_weight * remaining_prefill
            - length_weight * predicted_remaining_decode
            + self.slo_weight * urgency
        )


class PolicyFactory:

    _POLICY_REGISTRY = {
        "fcfs": FCFS,
        "sjf": SJF,
        "ljf": LJF,
        "srpt": SRPT,
        "srtf": SRPT,
        "prompt_len_sjf": PromptLenSJF,
        "priority": Priority,
        "edf": EDF,
        "least_laxity_first": LeastLaxityFirst,
        "aging_sjf": AgingSJF,
        "aging_srpt": AgingSRPT,
        "skip_join_mlfq": SkipJoinMLFQ,
        "prefill_aging_fairness": PrefillAgingFairness,
        "vtc_fairness": VTCFairness,
        "decode_first": DecodeFirst,
        "prefill_first": PrefillFirst,
        "memory_aware_sjf": MemoryAwareSJF,
        "hybrid": HybridAdaptive,
        "adaptive_io_aging": AdaptiveInputOutputAging,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
