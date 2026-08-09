#!/usr/bin/env python
"""
Aggregates results across all vllm_cpu_offloading sweep run directories into
one summary CSV: one row per (model, gpu_memory_utilization, chunked_prefill,
kv_offloading) combination.

Sources per run directory:
  - run_summary.txt / run.log : end-to-end processing time
  - results.csv               : p99 TTFT / TPOT / request latency
  - prometheus_metrics.jsonl  : preemption count, KV offload bytes (both
                                 directions), tokens served from offload cache
  - nvidia-smi-output.csv     : peak GPU memory used

Usage:
    python vllm-sarathi-bench/aggregate_results_for_vllm_sarathi.py \
        --root /home/sabiha/sarathi-serve/benchmark_output/vllm_cpu_offloading \
        --out vllm-sarathi-bench/sweep_summary.csv
"""
import argparse
import csv
import json
import os
import re

import numpy as np
import pandas as pd

RUN_NAME_RE = re.compile(
    r"^(?P<model>.+)_gpu_(?P<gpu_label>[0-9_]+)_(?P<chunk>chunk_on|chunk_off)_(?P<offload>offload_on|offload_off)$"
)
TIMING_LOG_RE = re.compile(r"end_to_end_s=([0-9.]+)")


def gpu_label_to_util(label):
    # "8_5" -> 0.85, "2_5" -> 0.25
    return float(label.replace("_", ".")) / 10.0


def parse_run_name(dir_name):
    m = RUN_NAME_RE.match(dir_name)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "gpu_memory_utilization": gpu_label_to_util(m.group("gpu_label")),
        "chunked_prefill": m.group("chunk") == "chunk_on",
        "kv_offloading": m.group("offload") == "offload_on",
    }


def get_end_to_end_s(run_dir):
    summary_path = os.path.join(run_dir, "run_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for line in f:
                if line.startswith("processing_end_to_end_s="):
                    return float(line.strip().split("=", 1)[1])

    log_path = os.path.join(run_dir, "run.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "[TIMING]" in line and "end_to_end_s=" in line:
                    m = TIMING_LOG_RE.search(line)
                    if m:
                        return float(m.group(1))
    return None


def get_percentiles(run_dir):
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    tpot = (df["latency_s"] - df["ttft_s"]) / (df["generated_tokens"] - 1)
    tpot = tpot.where(df["generated_tokens"] > 1)

    def p99(series):
        s = series.dropna()
        return float(np.percentile(s, 99)) if len(s) else None

    return {
        "num_requests": len(df),
        "p99_ttft_s": p99(df["ttft_s"]),
        "p99_tpot_s": p99(tpot),
        "p99_latency_s": p99(df["latency_s"]),
    }


def get_prometheus_stats(run_dir):
    path = os.path.join(run_dir, "prometheus_metrics.jsonl")
    if not os.path.exists(path):
        return {}

    last_line = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        return {}

    snap = json.loads(last_line)
    metrics = snap["metrics"]

    def find(name, label_key=None, label_val=None):
        for m in metrics:
            if m["name"] != name:
                continue
            if label_key is not None:
                actual = m.get("labels", {}).get(label_key)
                if actual is None or actual.lower() != label_val.lower():
                    continue
            return m
        return None

    out = {}

    preempt = find("vllm:num_preemptions")
    out["num_preemptions"] = preempt["value"] if preempt else None

    gpu_to_cpu = find("vllm:kv_offload_total_bytes", "transfer_type", "gpu_to_cpu")
    out["kv_offload_gpu_to_cpu_mb"] = (
        gpu_to_cpu["value"] / 1e6 if gpu_to_cpu else None
    )

    cpu_to_gpu = find("vllm:kv_offload_total_bytes", "transfer_type", "cpu_to_gpu")
    out["kv_offload_cpu_to_gpu_mb"] = (
        cpu_to_gpu["value"] / 1e6 if cpu_to_gpu else None
    )

    ext_transfer = find(
        "vllm:prompt_tokens_by_source", "source", "external_kv_transfer"
    )
    out["tokens_served_from_offload_cache"] = ext_transfer["value"] if ext_transfer else None

    return out


def get_max_gpu_mem_mib(run_dir):
    path = os.path.join(run_dir, "nvidia-smi-output.csv")
    if not os.path.exists(path):
        return None

    max_mem = None
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            mem_str = row[2].strip().replace(" MiB", "")
            try:
                mem = float(mem_str)
            except ValueError:
                continue
            if max_mem is None or mem > max_mem:
                max_mem = mem
    return max_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/sabiha/sarathi-serve/benchmark_output/vllm_cpu_offloading",
    )
    parser.add_argument("--out", default="sweep_summary.csv")
    args = parser.parse_args()

    rows = []
    for entry in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, entry)
        if not os.path.isdir(run_dir):
            continue

        is_failed = "_failed_" in entry
        base_name = entry.split("_failed_")[0] if is_failed else entry

        parsed = parse_run_name(base_name)
        if parsed is None:
            print(f"[SKIP] Unrecognized directory name: {entry}")
            continue

        row = dict(parsed)
        row["status"] = "failed" if is_failed else "ok"
        row["end_to_end_s"] = get_end_to_end_s(run_dir)
        row.update(get_percentiles(run_dir))
        row.update(get_prometheus_stats(run_dir))
        row["max_gpu_mem_mib"] = get_max_gpu_mem_mib(run_dir)

        rows.append(row)

    if not rows:
        print("No runs found.")
        return

    fieldnames = [
        "model", "gpu_memory_utilization", "chunked_prefill", "kv_offloading",
        "status", "end_to_end_s", "num_requests",
        "p99_ttft_s", "p99_tpot_s", "p99_latency_s",
        "max_gpu_mem_mib", "num_preemptions",
        "kv_offload_gpu_to_cpu_mb", "kv_offload_cpu_to_gpu_mb",
        "tokens_served_from_offload_cache",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
