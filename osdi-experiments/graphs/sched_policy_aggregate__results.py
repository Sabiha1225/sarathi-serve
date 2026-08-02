#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

# python osdi-experiments/graphs/sched_policy_aggregate__results.py

REPO = Path("/home/sabiha/sarathi_observation2")
TIMES_CSV = Path("/home/sabiha/sarathi_observation2/log/sched_policy_time.csv")
OUTPUT_ROOT = REPO / "benchmark_output" / "figure-1"
OUT_CSV = REPO / "osdi-experiments" / "graphs" / "sched_policy_full_results.csv"

CHUNK_SIZE = 512
QPS = 3
NUM_REQUESTS = 300

# maps the full HF model name (as it appears in sched_policy_time.csv) to the
# short label used in the sweep script's output directory names
MODEL_NAME_TO_SAFE = {
    "meta-llama/Llama-2-7b-hf": "llama2_7b",
    "01-ai/Yi-6B": "yi_6b",
    "DiscoResearch/mixtral-7b-8expert": "mixtral_7b_8expert",
}


def get_run_dir(model_name: str, dataset: str, policy: str) -> Path:
    model_safe = MODEL_NAME_TO_SAFE.get(model_name)
    if model_safe is None:
        raise ValueError(f"Unknown model_name '{model_name}', add it to MODEL_NAME_TO_SAFE")
    run_name = f"{model_safe}_{dataset}_sched_policy_{policy}_chunk{CHUNK_SIZE}_qps{QPS}_n{NUM_REQUESTS}"
    return OUTPUT_ROOT / run_name


def get_metrics_summary_fields(run_dir: Path) -> dict:
    path = run_dir / "replica_0" / "metrics_summary.json"
    if not path.exists():
        return {
            "ttft_p99": None,
            "tpot_p99": None,
            "request_latency_p99": None,
            "request_throughput_req_per_s": None,
            "token_throughput_tokens_per_s": None,
            "output_token_throughput_tokens_per_s": None,
        }

    with open(path) as f:
        summary = json.load(f)

    return {
        "ttft_p99": summary.get("ttft", {}).get("p99"),
        "tpot_p99": summary.get("tpot", {}).get("p99"),
        "request_latency_p99": summary.get("request_latency", {}).get("p99"),
        "request_throughput_req_per_s": summary.get("throughput", {}).get("request_throughput_req_per_s"),
        "token_throughput_tokens_per_s": summary.get("throughput", {}).get("token_throughput_tokens_per_s"),
        "output_token_throughput_tokens_per_s": summary.get("throughput", {}).get("output_token_throughput_tokens_per_s"),
    }


def get_scheduling_delay_p99(run_dir: Path):
    path = run_dir / "replica_0" / "preemption_metrics.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "scheduling_delay" not in df.columns or df["scheduling_delay"].dropna().empty:
        return None

    return float(np.percentile(df["scheduling_delay"].dropna(), 99))


def get_waiting_preemption_count(run_dir: Path):
    path = run_dir / "replica_0" / "preemption_details.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "state" not in df.columns:
        return None

    return int((df["state"] == "WAITING").sum())


def get_max_gpu_memory_mib(run_dir: Path):
    path = run_dir / "nvidia-smi-output.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "memory.used [MiB]" not in df.columns:
        return None

    mem_values = (
        df["memory.used [MiB]"]
        .astype(str)
        .str.strip()
        .str.replace(" MiB", "", regex=False)
        .astype(float)
    )
    if mem_values.empty:
        return None

    return float(mem_values.max())


def main():
    times_df = pd.read_csv(TIMES_CSV)

    rows = []
    for _, r in times_df.iterrows():
        model_name = r["model_name"]
        dataset = r["dataset"]
        policy = r["policy"]
        inference_time = r["inference_time"]

        run_dir = get_run_dir(model_name, dataset, policy)

        row = {
            "model_name": model_name,
            "dataset": dataset,
            "policy": policy,
            "inference_time": inference_time,
        }

        if not run_dir.exists():
            print(f"[MISSING RUN DIR] {run_dir}")
            row.update({
                "ttft_p99": None, "tpot_p99": None, "request_latency_p99": None,
                "request_throughput_req_per_s": None, "token_throughput_tokens_per_s": None,
                "output_token_throughput_tokens_per_s": None,
                "scheduling_delay_p99": None, "preemption_count_waiting": None,
                "max_gpu_memory_mib": None,
            })
            rows.append(row)
            continue

        row.update(get_metrics_summary_fields(run_dir))
        row["scheduling_delay_p99"] = get_scheduling_delay_p99(run_dir)
        row["preemption_count_waiting"] = get_waiting_preemption_count(run_dir)
        row["max_gpu_memory_mib"] = get_max_gpu_memory_mib(run_dir)

        rows.append(row)
        print(f"[OK] {model_name} / {dataset} / {policy}")

    fieldnames = [
        "model_name", "dataset", "policy", "inference_time",
        "ttft_p99", "tpot_p99", "request_latency_p99",
        "request_throughput_req_per_s", "token_throughput_tokens_per_s",
        "output_token_throughput_tokens_per_s",
        "scheduling_delay_p99", "preemption_count_waiting", "max_gpu_memory_mib",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
