#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

# python osdi-experiments/graphs/aggregate_arxiv_sarathi_results.py

REPO = Path("/home/sabiha/sarathi-serve")
TIMES_CSV = Path("/home/sabiha/sarathi-serve/log/time_arxiv_sarathi.csv")
OUTPUT_ROOT = REPO / "benchmark_output" / "arxiv"
OUT_CSV = REPO / "osdi-experiments" / "graphs" / "arxiv_sarathi_full_results.csv"


def get_run_dir(model: str, chunk_size: str, qps: str) -> Path:
    run_name = f"arxiv_chunk_{chunk_size}_poison_{qps}_qps_{model}"
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
        model = r["Model"]
        chunk_size = str(r["Chunk Size"])
        inference_time = str(r["Inference"]).replace(" s", "")
        qps = str(r["Request Arrival"]).split()[1]  # "Poisson 1 QPS" -> "1"

        run_dir = get_run_dir(model, chunk_size, qps)

        row = {
            "model": model,
            "chunk_size": chunk_size,
            "qps": qps,
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
        print(f"[OK] {model} / chunk={chunk_size} / qps={qps}")

    fieldnames = [
        "model", "chunk_size", "qps", "inference_time",
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
