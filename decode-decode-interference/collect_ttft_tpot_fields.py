#!/usr/bin/env python3
"""
Collects per-request fields from sequence_metrics.csv + preemption_metrics.csv
across both schedulers and all three models, into two CSVs:
  decode_decode_single_request_fields.csv  (5 reps x 3 models x 2 schedulers)
  decode_decode_mixed_request_fields.csv   (19 requests x 3 models x 2 schedulers)

python3 decode-decode-interference/collect_ttft_tpot_fields.py

"""
from pathlib import Path

import pandas as pd

REPO = Path("/home/sabiha/sarathi-serve")
SCHEDULERS = {
    "sarathi": REPO / "benchmark_output/sarathi_decode_decode_interference",
    "vllm": REPO / "benchmark_output/vllm_decode_decode_interference",
}
MODELS = ["llama", "mistral", "mixtral"]
NUM_REPS = 5

SEQ_COLS = [
    "Request Id",
    "request_num_prefill_tokens",
    "request_num_decode_tokens",
    "request_scheduling_delay",
    "request_execution_time",
    "request_e2e_time",
    "prefill_e2e_time",
    "decode_time_execution_plus_preemption_normalized",
]
PREEMPT_COLS = [
    "Request Id",
    "prefill_exec_with_preemption",
    "decode_exec_with_preemption",
]


def load_run(run_dir: Path) -> pd.DataFrame:
    seq = pd.read_csv(run_dir / "replica_0" / "sequence_metrics.csv")[SEQ_COLS]
    preempt = pd.read_csv(run_dir / "replica_0" / "preemption_metrics.csv")[PREEMPT_COLS]
    merged = seq.merge(preempt, on="Request Id", how="left", validate="one_to_one")

    merged = merged.rename(columns={
        "prefill_e2e_time": "ttft",
        "decode_time_execution_plus_preemption_normalized": "tpot",
    })

    return merged



# ---------------------------------------------------------------- single-request
single_rows = []
for scheduler, root in SCHEDULERS.items():
    for model in MODELS:
        for rep in range(1, NUM_REPS + 1):
            run_dir = root / f"{model}_single_req_{rep}"
            df = load_run(run_dir)
            df.insert(0, "scheduler", scheduler)
            df.insert(1, "model", model)
            df.insert(2, "rep", rep)
            single_rows.append(df)

single_df = pd.concat(single_rows, ignore_index=True)
single_df.to_csv(REPO / "decode-decode-interference" / "decode_decode_single_request_fields.csv", index=False)
print(f"wrote decode_decode_single_request_fields.csv ({len(single_df)} rows)")

# ---------------------------------------------------------------- mixed-request
mixed_rows = []
for scheduler, root in SCHEDULERS.items():
    for model in MODELS:
        run_dir = root / f"{model}_mixed_req"
        df = load_run(run_dir)
        df.insert(0, "scheduler", scheduler)
        df.insert(1, "model", model)
        mixed_rows.append(df)

mixed_df = pd.concat(mixed_rows, ignore_index=True)
mixed_df.to_csv(REPO / "decode-decode-interference" / "decode_decode_mixed_request_fields.csv", index=False)
print(f"wrote decode_decode_mixed_request_fields.csv ({len(mixed_df)} rows)")
