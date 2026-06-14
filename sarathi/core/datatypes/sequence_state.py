import time
from typing import List, Optional

from sarathi.core.datatypes.sequence_status import SequenceStatus

from dataclasses import dataclass, field

@dataclass
class PreemptionPhaseRecord:
    """Record detailed information about a single preemption event."""
    preemption_id: int  # Sequential ID of preemption event
    phase: str  # "PREFILL" or "DECODE"
    state: str # "Paused", "Waiting"
    preempt_time: float  # When preempted
    resume_time: Optional[float] = None  # When resumed (None if still preempted)
    preemption_duration: Optional[float] = None  # How long preempted
    
    # State at time of preemption
    prompt_tokens_processed: int = 0
    output_tokens_at_preemption: int = 0
    memory_bytes: int = 0
    num_blocks_allocated: int = 0
    
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "preemption_id": self.preemption_id,
            "phase": self.phase,
            "state": self.state,
            "preempt_time": self.preempt_time,
            "resume_time": self.resume_time,
            "preemption_duration_sec": self.preemption_duration,
            "prompt_tokens_processed": self.prompt_tokens_processed,
            "output_tokens_at_preemption": self.output_tokens_at_preemption,
            "total_tokens_at_preemption": self.prompt_tokens_processed + self.output_tokens_at_preemption,
            "memory_bytes": self.memory_bytes,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "num_blocks_allocated": self.num_blocks_allocated,
        }


class SequenceState:

    def __init__(self, id: str, arrived_at: float, num_prompt_tokens: int):
        self._id = id
        self._arrived_at: float = arrived_at
        self._num_prompt_tokens: int = num_prompt_tokens
        self._num_output_tokens: int = 0
        self._status = SequenceStatus.WAITING
        self._is_scheduled: bool = False
        self._is_completed: bool = False
        self._scheduled_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._prompt_processing_completed_at: Optional[float] = None
        self._last_restart_at: Optional[float] = None
        self._last_pause_at: Optional[float] = None
        self._execution_time: float = 0.0
        self._preempted_time: float = 0.0
        self._last_execution_start_at: Optional[float] = None
        self._num_restarts: int = 0
        self._num_pauses: int = 0
        self._is_ignore_finished: bool = False
        self._last_token_generated_at: Optional[float] = None
        self._last_token_generation_time: float = 0.0

        # ADD: Detailed preemption tracking
        self._preemption_records: List[PreemptionPhaseRecord] = []
        self._current_preemption_record: Optional[PreemptionPhaseRecord] = None
        self.phase = "PREFILL"
        self._preemption_counter: int = 0

        self._last_offloaded_at: float = 0.0

    @property
    def preemption_records(self) -> List[PreemptionPhaseRecord]:
        """Get all preemption records for this sequence."""
        return self._preemption_records

    @property
    def total_preemption_events(self) -> int:
        """Get total number of preemption events."""
        return len(self._preemption_records)

    @property
    def preemption_records_dict(self) -> List[dict]:
        """Convert all preemption records to dictionaries."""
        return [rec.to_dict() for rec in self._preemption_records]
    
    @property
    def total_prefill_preemptions(self) -> int:
        """Count preemptions during prefill phase."""
        return sum(1 for rec in self._preemption_records if rec.phase == "PREFILL")

    @property
    def total_decode_preemptions(self) -> int:
        """Count preemptions during decode phase."""
        return sum(1 for rec in self._preemption_records if rec.phase == "DECODE")

    @property
    def total_prefill_preemption_time(self) -> float:
        """Total time preempted during prefill."""
        return sum(
            rec.preemption_duration for rec in self._preemption_records 
            if rec.phase == "PREFILL" and rec.preemption_duration is not None
        )
    
    @property
    def total_decode_preemption_time(self) -> float:
        """Total time preempted during decode."""
        return sum(
            rec.preemption_duration for rec in self._preemption_records 
            if rec.phase == "DECODE" and rec.preemption_duration is not None
        )

    def record_preemption_start(
        self,
        prompt_tokens_processed: int,
        output_tokens: int,
        memory_bytes: int,
        num_blocks: int,
        state: str,
    ) -> None:
        """
        Record the start of a preemption event.
        
        Args:
            is_in_decode_phase: True if preempted during decode, False if during prefill
            prompt_tokens_processed: Number of prompt tokens processed so far
            output_tokens: Number of output tokens generated
            memory_bytes: Memory used by this sequence
            num_blocks: Number of blocks allocated
        """
        # phase = "DECODE" if is_in_decode_phase else "PREFILL"
        
        record = PreemptionPhaseRecord(
            preemption_id=self._preemption_counter,
            phase=self.phase,
            state=state,
            preempt_time=time.monotonic(),
            prompt_tokens_processed=prompt_tokens_processed,
            output_tokens_at_preemption=output_tokens,
            memory_bytes=memory_bytes,
            num_blocks_allocated=num_blocks,
        )
        
        self._current_preemption_record = record
        self._preemption_counter += 1

    def record_preemption_end(self) -> None:
        """Record the end of a preemption event."""
        if self._current_preemption_record is None:
            return
        
        current_time = time.monotonic()
        self._current_preemption_record.resume_time = current_time
        self._current_preemption_record.preemption_duration = (
            current_time - self._current_preemption_record.preempt_time
        )
        # self._current_preemption_record.decode_tokens_added_before_resume = (
        #     self._decode_tokens_added_during_preemption
        # )
        
        self._preemption_records.append(self._current_preemption_record)
        self._current_preemption_record = None

    def get_preemption_summary(self) -> dict:
        """Get comprehensive preemption summary for this sequence."""
        return {
            "total_preemption_events": self.total_preemption_events,
            "prefill_preemptions": self.total_prefill_preemptions,
            "decode_preemptions": self.total_decode_preemptions,
            "total_prefill_preemption_time_sec": self.total_prefill_preemption_time,
            "total_decode_preemption_time_sec": self.total_decode_preemption_time,
            "total_preemption_time_sec": self.total_prefill_preemption_time + self.total_decode_preemption_time,
            "preemption_records": self.preemption_records_dict,
        }

    @property
    def id(self) -> str:
        return self._id

    @property
    def num_prompt_tokens(self) -> int:
        return self._num_prompt_tokens

    @property
    def num_output_tokens(self) -> int:
        return self._num_output_tokens

    @property
    def num_total_tokens(self) -> int:
        return self._num_prompt_tokens + self._num_output_tokens

    @property
    def status(self) -> SequenceStatus:
        return self._status

    @property
    def is_scheduled(self) -> bool:
        return self._is_scheduled

    @property
    def is_completed(self) -> bool:
        return self._is_completed

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def scheduled_at(self) -> Optional[float]:
        return self._scheduled_at

    @property
    def completed_at(self) -> Optional[float]:
        return self._completed_at

    @property
    def prompt_processing_completed_at(self) -> Optional[float]:
        return self._prompt_processing_completed_at

    @property
    def e2e_time(self) -> Optional[float]:
        return (
            self._completed_at - self._arrived_at
            if self._completed_at is not None
            else None
        )

    @property
    def e2e_time_piecewise_normalized(self) -> float:
        return self.scheduling_delay + (
            self.execution_plus_preemption_time / self._num_output_tokens
        )

    @property
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self._num_output_tokens

    @property
    def e2e_prefill_time(self) -> Optional[float]:
        return (
            self._prompt_processing_completed_at - self._arrived_at
            if self._prompt_processing_completed_at is not None
            else None
        )

    @property
    def e2e_prefill_time_normalized(self) -> Optional[float]:
        return (
            (self.e2e_prefill_time / self._num_prompt_tokens)
            if self._prompt_processing_completed_at is not None
            else None
        )

    @property
    def e2e_prefill_time_piecewise_normalized(self) -> Optional[float]:
        return (
            self.scheduling_delay
            + (self.prefill_execution_plus_preemption_time / self._num_prompt_tokens)
            if self._prompt_processing_completed_at
            else None
        )

    @property
    def prefill_execution_plus_preemption_time(self) -> float:
        return (
            self._prompt_processing_completed_at - self._scheduled_at
            if self._prompt_processing_completed_at is not None
            else None
        )

    @property
    def decode_execution_plus_preemption_time(self) -> float:
        return (
            self._completed_at - self._prompt_processing_completed_at
            if self._completed_at is not None
            else None
        )

    @property
    def prefill_execution_plus_preemption_time_normalized(self) -> Optional[float]:
        return (
            self.prefill_execution_plus_preemption_time / self._num_prompt_tokens
            if self.prefill_execution_plus_preemption_time
            else None
        )

    @property
    def decode_execution_plus_preemption_time_normalized(self) -> Optional[float]:
        return (
            self.decode_execution_plus_preemption_time / self._num_output_tokens
            if self.decode_execution_plus_preemption_time
            else None
        )

    @property
    def scheduling_delay(self) -> Optional[float]:
        return (
            self._scheduled_at - self._arrived_at
            if self._scheduled_at is not None
            else None
        )

    @property
    def execution_time(self) -> float:
        return self._execution_time

    @property
    def execution_time_normalized(self) -> float:
        return self.execution_time / self._num_output_tokens

    @property
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    def execution_plus_preemption_time(self) -> float:
        return self._execution_time + self._preempted_time

    @property
    def execution_plus_preemption_time_normalized(self) -> float:
        return self.execution_plus_preemption_time / self._num_output_tokens

    @property
    def last_token_generation_time(self) -> float:
        return self._last_token_generation_time

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def num_pauses(self) -> int:
        return self._num_pauses

    @property
    def is_ignore_finished(self) -> bool:
        return self._is_ignore_finished

    def _handle_transitions_from_waiting_status(
        self, current_time: float, status: SequenceStatus
    ) -> None:
        if status == SequenceStatus.RUNNING:
            # request is starting execution now
            if self._scheduled_at is None:
                # running for the first time
                assert self._num_restarts == 0
                self._is_scheduled = True
                self._scheduled_at = current_time
            else:
                # restarting
                assert self._num_restarts > 0
                self._preempted_time += current_time - self._last_restart_at

            self._last_execution_start_at = current_time
        elif status == SequenceStatus.FINISHED_IGNORED:
            self._is_ignore_finished = True
            self._is_completed = True
            self._completed_at = current_time
            # the scheduler will not schedule this request again
            self._scheduled_at = current_time
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_running_status(
        self, current_time: float, status: SequenceStatus
    ) -> None:
        self._execution_time += current_time - self._last_execution_start_at

        if status == SequenceStatus.PAUSED:
            self._num_pauses += 1
            self._last_pause_at = current_time
        elif status == SequenceStatus.WAITING:
            self._num_restarts += 1
            self._last_restart_at = current_time
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_paused_status(
        self, current_time: float, status: SequenceStatus
    ) -> None:
        self._preempted_time += current_time - self._last_pause_at

        if (
            status == SequenceStatus.FINISHED_STOPPED
            or status == SequenceStatus.FINISHED_LENGTH_CAPPED
        ):
            self._is_completed = True
            self._completed_at = current_time
        elif status == SequenceStatus.RUNNING:
            self._last_execution_start_at = current_time
        elif status == SequenceStatus.WAITING:
            self._num_restarts += 1
            self._last_restart_at = current_time
        elif status == SequenceStatus.OFFLOADED:
            self._last_offloaded_at = current_time
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )
    def _handle_transitions_from_offloaded_status(
        self, current_time: float, status: SequenceStatus
    ) -> None:
        if status == SequenceStatus.RUNNING:
            self._preempted_time += current_time - self._last_offloaded_at
            self._last_execution_start_at = current_time
        elif (
            status == SequenceStatus.FINISHED_STOPPED
            or status == SequenceStatus.FINISHED_LENGTH_CAPPED
        ):
            self._preempted_time += current_time - self._last_offloaded_at
            self._is_completed = True
            self._completed_at = current_time
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def set_status(self, status: SequenceStatus) -> None:
        current_time = time.monotonic()

        if self._status == SequenceStatus.WAITING:
            self._handle_transitions_from_waiting_status(current_time, status)
        elif self._status == SequenceStatus.RUNNING:
            self._handle_transitions_from_running_status(current_time, status)
        elif self._status == SequenceStatus.PAUSED:
            self._handle_transitions_from_paused_status(current_time, status)
        elif self._status == SequenceStatus.OFFLOADED:
            self._handle_transitions_from_offloaded_status(current_time, status)
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

        self._status = status

    def on_prompt_processing_completed(self) -> None:
        self._prompt_processing_completed_at = time.monotonic()
        self.phase = "DECODE"

    def on_token_generated(self) -> None:
        current_time = time.monotonic()

        self._num_output_tokens += 1

        if not self._last_token_generated_at:
            self._last_token_generation_time = 0
        else:
            self._last_token_generation_time = (
                current_time - self._last_token_generated_at
            )

        self._last_token_generated_at = current_time
