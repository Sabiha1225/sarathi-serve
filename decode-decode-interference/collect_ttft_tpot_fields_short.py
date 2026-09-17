#!/usr/bin/env python3
"""
Collects per-request fields from sequence_metrics.csv + preemption_metrics.csv
for the all-short mixed-request control runs, across both schedulers and all
three models, into two CSVs (one per arrival mode):
  decode_decode_mixed_request_short_static_fields.csv       (19 requests x 3 models x 2 schedulers)
  decode_decode_mixed_request_short_poisson_qps2_fields.csv (19 requests x 3 models x 2 schedulers)

python3 decode-decode-interference/collect_ttft_tpot_fields_short.py
"""
from pathlib import Path

import pandas as pd

REPO = Path("/home/sabiha/sarathi-serve")
SCHEDULERS = {
    "sarathi": REPO / "benchmark_output/sarathi_decode_decode_interference",
    "vllm": REPO / "benchmark_output/vllm_decode_decode_interference",
}
MODELS = ["llama", "mistral", "mixtral"]
ARRIVAL_MODES = ["static", "poisson_qps2"]

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


for mode in ARRIVAL_MODES:
    rows = []
    for scheduler, root in SCHEDULERS.items():
        for model in MODELS:
            run_dir = root / f"{model}_mixed_req_short_{mode}"
            df = load_run(run_dir)
            df.insert(0, "scheduler", scheduler)
            df.insert(1, "model", model)
            rows.append(df)

    out_df = pd.concat(rows, ignore_index=True)
    out_path = REPO / "decode-decode-interference" / f"decode_decode_mixed_request_short_{mode}_fields.csv"
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out_df)} rows)")
