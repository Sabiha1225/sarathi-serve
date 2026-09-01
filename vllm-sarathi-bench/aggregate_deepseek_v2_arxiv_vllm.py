#!/usr/bin/env python3

"""
Aggregate non-chunked DeepSeek-V2 ARXIV vLLM results.

Expected directory format:
    deepseek_v2_lite_qps_1_chunk_off_offload_off
    deepseek_v2_lite_qps_5_chunk_off_offload_off
    deepseek_v2_lite_qps_10_chunk_off_offload_off
    deepseek_v2_lite_qps_15_chunk_off_offload_off
    deepseek_v2_lite_qps_20_chunk_off_offload_off

The script produces one row per QPS value.


python -u \
  vllm-sarathi-bench/aggregate_deepseek_v2_arxiv_vllm.py \
  --root benchmark_output/vllm_deepseek_v2_arxiv_chunk_off \
  --out vllm-sarathi-bench/deepseek_aggregated_results_vllm.csv

"""

import argparse
import csv
import re
from pathlib import Path

from aggregate_deepseek_v2_arxiv import (
    get_end_to_end_s,
    get_max_gpu_memory_mib,
    get_num_preemptions,
    get_request_statistics,
    read_config,
    safe_float,
)


DEFAULT_ROOT = Path(
    "/home/sabiha/sarathi-serve/"
    "benchmark_output/vllm_deepseek_v2_arxiv_chunk_off"
)

RUN_NAME_RE = re.compile(
    r"^deepseek_v2_lite"
    r"_qps_(?P<qps>[0-9.]+)"
    r"_chunk_off"
    r"_offload_off"
    r"(?:_failed_.*)?$"
)


def parse_qps_from_name(directory_name):
    """Read QPS from the run-directory name."""
    match = RUN_NAME_RE.match(directory_name)

    if not match:
        return None

    return safe_float(match.group("qps"))


def build_row(run_dir):
    """Build one aggregate row from a benchmark run."""
    config = read_config(run_dir)

    model = config.get("model", {})
    scheduler = config.get("vllm_scheduler", {})
    request_config = config.get("request_generator", {})
    interval_config = config.get(
        "poisson_request_interval_generator",
        {},
    )
    offloading = config.get("offloading", {})

    qps = safe_float(interval_config.get("qps"))

    if qps is None:
        qps = parse_qps_from_name(run_dir.name)

    request_stats = get_request_statistics(run_dir)
    end_to_end_s = get_end_to_end_s(run_dir)

    has_results = (run_dir / "results.csv").exists()

    status = (
        "success"
        if has_results and end_to_end_s is not None
        else "incomplete_or_failed"
    )

    kv_offloading = (
        offloading.get("kv_offloading_size") is not None
    )

    return {
        "model_name": model.get("name"),
        "configured_num_requests": request_config.get(
            "num_requests"
        ),
        "num_requests": request_stats["num_requests"],
        "qps": qps,
        "chunked_prefill": scheduler.get(
            "enable_chunked_prefill",
            False,
        ),
        "max_num_batched_tokens": scheduler.get(
            "max_num_batched_tokens"
        ),
        "tensor_parallel_size": model.get(
            "tensor_parallel_size"
        ),
        "pipeline_parallel_size": model.get(
            "pipeline_parallel_size"
        ),
        "kv_offloading": kv_offloading,
        "end_to_end_s": end_to_end_s,
        "p99_ttft_s": request_stats["p99_ttft_s"],
        "p99_tpot_s": request_stats["p99_tpot_s"],
        "p99_request_latency_s": request_stats[
            "p99_request_latency_s"
        ],
        "num_preemptions": get_num_preemptions(run_dir),
        "max_gpu_memory_mib": get_max_gpu_memory_mib(
            run_dir
        ),
        "status": status,
    }


def sort_key(row):
    """Sort output rows numerically by QPS."""
    qps = row["qps"]

    if qps is None:
        return float("inf")

    return float(qps)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing non-chunked runs",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output CSV. Default: "
            "<root>/deepseek_aggregated_results_vllm.csv"
        ),
    )

    args = parser.parse_args()

    root = args.root.resolve()

    output_path = (
        args.out.resolve()
        if args.out
        else root / "deepseek_aggregated_results_vllm.csv"
    )

    if not root.is_dir():
        raise SystemExit(
            f"Benchmark root does not exist: {root}"
        )

    rows = []

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        if not RUN_NAME_RE.match(run_dir.name):
            print(
                f"[SKIP] Unrecognized directory: "
                f"{run_dir.name}"
            )
            continue

        print(f"[READ] {run_dir.name}")
        rows.append(build_row(run_dir))

    if not rows:
        raise SystemExit(
            f"No non-chunked benchmark runs found under {root}"
        )

    rows.sort(key=sort_key)

    fieldnames = [
        "model_name",
        "configured_num_requests",
        "num_requests",
        "qps",
        "chunked_prefill",
        "max_num_batched_tokens",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "kv_offloading",
        "end_to_end_s",
        "p99_ttft_s",
        "p99_tpot_s",
        "p99_request_latency_s",
        "num_preemptions",
        "max_gpu_memory_mib",
        "status",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(
        row["status"] == "success"
        for row in rows
    )

    print()
    print(f"Wrote {len(rows)} rows to:")
    print(output_path)
    print(f"Successful runs: {successful}")
    print(
        "Incomplete/failed runs: "
        f"{len(rows) - successful}"
    )


if __name__ == "__main__":
    main()