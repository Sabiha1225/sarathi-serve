#!/usr/bin/env python3

"""
Aggregate DeepSeek-V2 ARXIV vLLM sweep results.

One output row is generated for each QPS/chunk-size run.

Sources:
  config.yml:
    model, configured requests, QPS, chunk size, TP, PP

  results.csv:
    actual number of completed requests
    p99 TTFT
    p99 TPOT
    p99 request latency

  run.log:
    request-processing end-to-end time

  prometheus_metrics.jsonl:
    cumulative number of preemptions

  nvidia-smi-output.csv:
    maximum memory used by any individual GPU

Usage:
    python -u \
      vllm-sarathi-bench/aggregate_deepseek_v2_arxiv.py
/home/sabiha/sarathi-serve/vllm-sarathi-bench
Or:

    python -u \
      vllm-sarathi-bench/aggregate_deepseek_v2_arxiv.py \
      --root benchmark_output/vllm_deepseek_v2_arxiv \
      --out vllm-sarathi-bench/deepseek_aggregated_results.csv
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_ROOT = Path(
    "/home/sabiha/sarathi-serve/"
    "benchmark_output/vllm_deepseek_v2_arxiv"
)

RUN_NAME_RE = re.compile(
    r"^deepseek_v2_lite"
    r"_qps_(?P<qps>[0-9.]+)"
    r"_chunk_(?P<chunk>[0-9]+)"
    r"_offload_off"
    r"(?:_failed_.*)?$"
)

END_TO_END_RE = re.compile(
    r"end_to_end_s=(?P<value>[0-9]+(?:\.[0-9]+)?)"
)


def safe_float(value):
    """Convert a value to float, returning None when unavailable."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def percentile_99(series):
    """Calculate p99 after removing missing and non-finite values."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    if numeric.empty:
        return None

    return float(np.percentile(numeric.to_numpy(), 99))


def read_config(run_dir):
    """Read a run's generated config.yml."""
    path = run_dir / "config.yml"

    if not path.exists():
        return {}

    try:
        with path.open() as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return {}

    return config if isinstance(config, dict) else {}


def parse_run_name(run_name):
    """Fallback parser when values are missing from config.yml."""
    match = RUN_NAME_RE.match(run_name)

    if not match:
        return {}

    return {
        "qps": safe_float(match.group("qps")),
        "chunk_size": int(match.group("chunk")),
    }


def get_end_to_end_s(run_dir):
    """
    Read request-processing end-to-end time from run.log.

    Expected line:
      [TIMING] ... -- end_to_end_s=1281.697
    """
    path = run_dir / "run.log"

    if not path.exists():
        return None

    last_value = None

    try:
        with path.open(errors="replace") as f:
            for line in f:
                match = END_TO_END_RE.search(line)
                if match:
                    last_value = float(match.group("value"))
    except OSError as exc:
        print(f"[WARN] Cannot read {path}: {exc}")

    return last_value


def get_request_statistics(run_dir):
    """
    Calculate request count and client-observed latency percentiles.

    TPOT definition:
      (request_latency - TTFT) / (generated_tokens - 1)

    The first generated token is already represented by TTFT, so the
    denominator is generated_tokens - 1.
    """
    path = run_dir / "results.csv"

    empty = {
        "num_requests": None,
        "p99_ttft_s": None,
        "p99_tpot_s": None,
        "p99_request_latency_s": None,
    }

    if not path.exists():
        return empty

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return empty

    if df.empty:
        return {
            "num_requests": 0,
            "p99_ttft_s": None,
            "p99_tpot_s": None,
            "p99_request_latency_s": None,
        }

    required = {
        "ttft_s",
        "latency_s",
        "generated_tokens",
    }

    missing = required - set(df.columns)
    if missing:
        print(
            f"[WARN] {path} is missing columns: "
            f"{sorted(missing)}"
        )
        return {
            "num_requests": len(df),
            "p99_ttft_s": None,
            "p99_tpot_s": None,
            "p99_request_latency_s": None,
        }

    ttft = pd.to_numeric(df["ttft_s"], errors="coerce")
    latency = pd.to_numeric(df["latency_s"], errors="coerce")
    generated = pd.to_numeric(
        df["generated_tokens"],
        errors="coerce",
    )

    # TPOT is valid only when at least two tokens were generated.
    valid_tpot = generated > 1
    tpot = pd.Series(np.nan, index=df.index, dtype=float)

    tpot.loc[valid_tpot] = (
        latency.loc[valid_tpot] - ttft.loc[valid_tpot]
    ) / (
        generated.loc[valid_tpot] - 1
    )

    # Reject negative values caused by malformed timing records.
    tpot = tpot.where(tpot >= 0)

    return {
        "num_requests": int(len(df)),
        "p99_ttft_s": percentile_99(ttft),
        "p99_tpot_s": percentile_99(tpot),
        "p99_request_latency_s": percentile_99(latency),
    }


def get_num_preemptions(run_dir):
    """
    Read vllm:num_preemptions from Prometheus snapshots.

    The metric is cumulative. We take the maximum across every saved
    snapshot rather than only the last line. This is more robust when the
    final metrics snapshot was recorded a few seconds before completion.
    """
    path = run_dir / "prometheus_metrics.jsonl"

    if not path.exists():
        return None

    max_preemptions = None

    try:
        with path.open(errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    snapshot = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[WARN] Invalid JSON in {path} "
                        f"at line {line_number}"
                    )
                    continue

                metrics = snapshot.get("metrics", [])

                for metric in metrics:
                    if metric.get("name") != "vllm:num_preemptions":
                        continue

                    value = safe_float(metric.get("value"))

                    if value is None:
                        continue

                    if (
                        max_preemptions is None
                        or value > max_preemptions
                    ):
                        max_preemptions = value

    except OSError as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return None

    if max_preemptions is None:
        return None

    # The metric is a counter, so present integral values as integers.
    if max_preemptions.is_integer():
        return int(max_preemptions)

    return max_preemptions


def get_max_gpu_memory_mib(run_dir):
    """
    Return the maximum memory used by any individual GPU at any sample.

    Expected CSV columns:
      timestamp, index, memory.used [MiB], utilization.gpu [%]
    """
    path = run_dir / "nvidia-smi-output.csv"

    if not path.exists():
        return None

    maximum = None

    try:
        with path.open(errors="replace", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                return None

            memory_column = next(
                (
                    name for name in reader.fieldnames
                    if name and "memory.used" in name
                ),
                None,
            )

            if memory_column is None:
                print(
                    f"[WARN] No memory.used column in {path}"
                )
                return None

            for row in reader:
                raw = row.get(memory_column, "")
                raw = raw.replace("MiB", "").strip()
                value = safe_float(raw)

                if value is None:
                    continue

                if maximum is None or value > maximum:
                    maximum = value

    except OSError as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return None

    return maximum


def build_row(run_dir):
    """Aggregate one run directory."""
    config = read_config(run_dir)
    name_values = parse_run_name(run_dir.name)

    model = config.get("model", {})
    request_config = config.get("request_generator", {})
    interval_config = config.get(
        "poisson_request_interval_generator",
        {},
    )
    scheduler = config.get("vllm_scheduler", {})

    qps = safe_float(interval_config.get("qps"))
    if qps is None:
        qps = name_values.get("qps")

    chunk_size = scheduler.get("max_num_batched_tokens")
    if chunk_size is None:
        chunk_size = name_values.get("chunk_size")

    stats = get_request_statistics(run_dir)
    end_to_end_s = get_end_to_end_s(run_dir)

    has_results = (run_dir / "results.csv").exists()
    status = (
        "success"
        if has_results and end_to_end_s is not None
        else "incomplete_or_failed"
    )

    return {
        "model_name": model.get("name"),
        "configured_num_requests": request_config.get(
            "num_requests"
        ),
        "num_requests": stats["num_requests"],
        "qps": qps,
        "chunk_size": chunk_size,
        "tensor_parallel_size": model.get(
            "tensor_parallel_size"
        ),
        "pipeline_parallel_size": model.get(
            "pipeline_parallel_size"
        ),
        "end_to_end_s": end_to_end_s,
        "p99_ttft_s": stats["p99_ttft_s"],
        "p99_tpot_s": stats["p99_tpot_s"],
        "p99_request_latency_s": (
            stats["p99_request_latency_s"]
        ),
        "num_preemptions": get_num_preemptions(run_dir),
        "max_gpu_memory_mib": get_max_gpu_memory_mib(
            run_dir
        ),
        "status": status,
        # "run_directory": str(run_dir),
    }


def sort_key(row):
    """Sort numerically by QPS and chunk size."""
    qps = row["qps"]
    chunk = row["chunk_size"]

    return (
        float("inf") if qps is None else float(qps),
        float("inf") if chunk is None else int(chunk),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root containing one directory per sweep run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: "
            "<root>/deepseek_aggregated_results.csv"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = (
        args.out.resolve()
        if args.out
        else root / "deepseek_aggregated_results.csv"
    )

    if not root.is_dir():
        raise SystemExit(
            f"Benchmark root does not exist: {root}"
        )

    rows = []

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        # Require either a recognized directory name or config file.
        recognized_name = RUN_NAME_RE.match(run_dir.name)
        has_config = (run_dir / "config.yml").exists()

        if not recognized_name and not has_config:
            print(
                f"[SKIP] Unrecognized directory: {run_dir.name}"
            )
            continue

        print(f"[READ] {run_dir.name}")
        rows.append(build_row(run_dir))

    if not rows:
        raise SystemExit(
            f"No benchmark run directories found under {root}"
        )

    rows.sort(key=sort_key)

    fieldnames = [
        "model_name",
        "configured_num_requests",
        "num_requests",
        "qps",
        "chunk_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "end_to_end_s",
        "p99_ttft_s",
        "p99_tpot_s",
        "p99_request_latency_s",
        "num_preemptions",
        "max_gpu_memory_mib",
        "status",
        # "run_directory",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

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