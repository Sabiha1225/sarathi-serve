#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import pandas as pd
import yaml


# python scripts/run_custom_summary_csv.py \
#  --benchmark-root benchmark_output/figure-1 \
#  --log-root log \
#  --output-dir custom_data_output

REQUEST_TYPE_KEYS = [
    "long_input_long_output",
    "long_input_short_output",
    "mixed_long_short_input_output",
    "mixed_long_short_output",
    "short_input_long_output",
    "short_input_short_output",
]

REQUEST_TYPE_LABELS = {
    "long_input_long_output": "long input long output",
    "long_input_short_output": "long input short output",
    "mixed_long_short_input_output": "mixed long short input output",
    "mixed_long_short_output": "mixed long short input output",
    "short_input_long_output": "short input long output",
    "short_input_short_output": "short input short output",
}

REQUEST_TYPE_ORDER = [
    "long input long output",
    "long input short output",
    "mixed long short input output",
    "short input long output",
    "short input short output",
]

SCHEDULER_ORDER = {
    "sarathi": 0,
    "vllm": 1,
}


def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def short_model_name(model_name):
    name = str(model_name).split("/")[-1].lower()

    if "llama-2-7b" in name:
        return "llama2_7b"
    if "mistral-7b" in name:
        return "mistral_7b"
    if "mixtral" in name:
        return "mixtral_7b_8expert"
    if "qwen" in name:
        return "qwen_7b"
    if "yi" in name:
        return "yi_6b"

    return str(model_name).split("/")[-1].replace("-", "_").lower()


def pretty_request_type(request_type):
    return REQUEST_TYPE_LABELS.get(request_type, request_type.replace("_", " "))


def parse_custom_run_name(run_name):
    request_type_pattern = "|".join(REQUEST_TYPE_KEYS)

    pattern = re.compile(
        rf"^custom_(?P<scheduler>sarathi|vllm)_"
        rf"(?P<request_type>{request_type_pattern})_"
    )

    match = pattern.search(run_name)

    if not match:
        return None

    return {
        "scheduler": match.group("scheduler"),
        "input_output_data_type": pretty_request_type(match.group("request_type")),
    }


def parse_number(value):
    if pd.isna(value):
        return float("nan")

    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if match:
        return float(match.group(0))

    return float("nan")


def read_gpu_max_memory(run_dir):
    gpu_path = run_dir / "nvidia-smi-output.csv"

    if not gpu_path.exists():
        return float("nan")

    df = pd.read_csv(gpu_path)

    memory_col = None
    for col in df.columns:
        if "memory.used" in col:
            memory_col = col
            break

    if memory_col is None:
        return float("nan")

    return df[memory_col].map(parse_number).max()


def parse_time_logs(log_root):
    rows = []

    if log_root is None or not log_root.exists():
        return pd.DataFrame(rows)

    request_type_pattern = "|".join(REQUEST_TYPE_KEYS)

    filename_re = re.compile(
        rf"^custom_data_time_(?P<scheduler>sarathi|vllm)_"
        rf"(?P<request_type>{request_type_pattern})\.txt$"
    )

    meta_re = re.compile(
        r"^Custom\s+.*?:\s+"
        r"model=(?P<model>\S+)\s+"
        r"num_requests=(?P<num_requests>\d+)\s+"
        r"qps=(?P<qps>[\d.]+)"
    )

    time_re = re.compile(r"Total time taken:\s*(?P<seconds>[\d.]+)\s*seconds")

    for path in sorted(log_root.glob("custom_data_time_*.txt")):
        file_match = filename_re.match(path.name)

        if not file_match:
            continue

        scheduler = file_match.group("scheduler")
        input_output_data_type = pretty_request_type(file_match.group("request_type"))

        current = None

        with open(path, "r", errors="ignore") as f:
            for line in f:
                meta = meta_re.search(line)

                if meta:
                    current = {
                        "Scheduler": scheduler,
                        "Input Output data Type": input_output_data_type,
                        "Model Name": short_model_name(meta.group("model")),
                        "Number of requests": int(meta.group("num_requests")),
                        "QPS": float(meta.group("qps")),
                    }
                    continue

                timing = time_re.search(line)

                if timing and current:
                    row = dict(current)
                    row["Inference Time"] = float(timing.group("seconds"))
                    rows.append(row)
                    current = None

    if not rows:
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)

    return (
        df.groupby(
            [
                "Scheduler",
                "Input Output data Type",
                "Model Name",
                "Number of requests",
                "QPS",
            ],
            as_index=False,
            dropna=False,
        )
        .agg({"Inference Time": "mean"})
    )


def find_custom_run_dirs(benchmark_root):
    run_dirs = []

    for run_dir in benchmark_root.iterdir():
        if not run_dir.is_dir():
            continue

        if not run_dir.name.startswith("custom_"):
            continue

        if not parse_custom_run_name(run_dir.name):
            continue

        sequence_path = run_dir / "replica_0" / "sequence_metrics.csv"
        preemption_path = run_dir / "replica_0" / "preemption_metrics.csv"
        config_path = run_dir / "benchmark_config.yml"

        if sequence_path.exists() and preemption_path.exists() and config_path.exists():
            run_dirs.append(run_dir)

    return sorted(run_dirs)


def load_request_level_rows(run_dir):
    parsed_name = parse_custom_run_name(run_dir.name)
    config = read_yaml(run_dir / "benchmark_config.yml")

    scheduler = parsed_name["scheduler"]
    input_output_data_type = parsed_name["input_output_data_type"]
    model_name = short_model_name(config.get("model_name", "unknown"))
    number_of_requests = int(config.get("synthetic_request_generator_num_requests", 0))
    qps = float(config.get("poisson_request_interval_generator_qps", 0.0))

    sequence = pd.read_csv(run_dir / "replica_0" / "sequence_metrics.csv")
    preemption = pd.read_csv(run_dir / "replica_0" / "preemption_metrics.csv")

    sequence_cols = [
        "Request Id",
        "request_scheduling_delay",
        "request_e2e_time",
        "request_num_prefill_tokens",
        "request_num_decode_tokens",
    ]

    preemption_cols = [
        "Request Id",
        "execution_time_with_preemptions",
        "prefill_exec_with_preemption",
        "decode_exec_with_preemption",
    ]

    sequence = sequence[[col for col in sequence_cols if col in sequence.columns]]
    preemption = preemption[[col for col in preemption_cols if col in preemption.columns]]

    df = sequence.merge(preemption, on="Request Id", how="left")

    for col in df.columns:
        if col != "Request Id":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    output = pd.DataFrame(index=df.index)
    output["Scheduler"] = scheduler
    output["Input Output data Type"] = input_output_data_type
    output["Model Name"] = model_name
    output["Number of Requests"] = number_of_requests
    output["QPS"] = qps
    output["Request No"] = range(len(df))

    output["Prefill Token Count"] = df.get("request_num_prefill_tokens")
    output["Decode Token Count"] = df.get("request_num_decode_tokens")
    output["Prefill Time"] = df.get("prefill_exec_with_preemption")
    output["Decode Time"] = df.get("decode_exec_with_preemption")
    output["Request Execution Time"] = df.get("execution_time_with_preemptions")
    output["Request End-to-End time"] = df.get("request_e2e_time")
    output["Scheduling Delay"] = df.get("request_scheduling_delay")

    output["Run Directory"] = str(run_dir)
    output["Max GPU Memory"] = read_gpu_max_memory(run_dir)

    return output


def build_request_level_csv(benchmark_root):
    frames = []

    for run_dir in find_custom_run_dirs(benchmark_root):
        try:
            frames.append(load_request_level_rows(run_dir))
        except Exception as exc:
            print(f"WARN: skipping {run_dir}: {exc}")

    if not frames:
        raise RuntimeError(f"No valid custom benchmark directories found under {benchmark_root}")

    request_level = pd.concat(frames, ignore_index=True)

    request_level["request_type_order"] = request_level["Input Output data Type"].map(
        {name: i for i, name in enumerate(REQUEST_TYPE_ORDER)}
    )
    request_level["scheduler_order"] = request_level["Scheduler"].map(SCHEDULER_ORDER)

    request_level = request_level.sort_values(
        [
            "request_type_order",
            "scheduler_order",
            "Model Name",
            "Number of Requests",
            "QPS",
            "Request No",
        ],
        na_position="last",
    )

    request_level = request_level.drop(columns=["request_type_order", "scheduler_order"])

    return request_level


def p95(series):
    return pd.to_numeric(series, errors="coerce").dropna().quantile(0.95)


def build_summary_csv(request_level, time_logs):
    df = request_level.copy()

    df["TTFT"] = df["Prefill Time"]

    denominator = (df["Decode Token Count"] - 1).clip(lower=1)
    df["TPOT"] = df["Decode Time"] / denominator

    df["Total Token Count"] = df["Prefill Token Count"] + df["Decode Token Count"]

    grouped = (
        df.groupby(
            [
                "Scheduler",
                "Input Output data Type",
                "Model Name",
                "Number of Requests",
                "QPS",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            **{
                "P95 TTFT": ("TTFT", p95),
                "P95 TPOT": ("TPOT", p95),
                "Request Latency": ("Request End-to-End time", p95),
                "Total Tokens": ("Total Token Count", "sum"),
                "Output Tokens": ("Decode Token Count", "sum"),
                "Max GPU Memory": ("Max GPU Memory", "max"),
            }
        )
    )

    grouped = grouped.rename(columns={"Number of Requests": "Number of requests"})

    if not time_logs.empty:
        grouped = grouped.merge(
            time_logs,
            on=[
                "Scheduler",
                "Input Output data Type",
                "Model Name",
                "Number of requests",
                "QPS",
            ],
            how="left",
        )
    else:
        grouped["Inference Time"] = float("nan")

    grouped["Request Throughput (req/s)"] = (
        grouped["Number of requests"] / grouped["Inference Time"]
    )

    grouped["Token Throughput (tokens/s)"] = (
        grouped["Total Tokens"] / grouped["Inference Time"]
    )

    grouped["Output Token Throughput (output-tokens/s)"] = (
        grouped["Output Tokens"] / grouped["Inference Time"]
    )

    summary_cols = [
        "Scheduler",
        "Input Output data Type",
        "Model Name",
        "Number of requests",
        "QPS",
        "Inference Time",
        "P95 TTFT",
        "P95 TPOT",
        "Request Latency",
        "Request Throughput (req/s)",
        "Token Throughput (tokens/s)",
        "Output Token Throughput (output-tokens/s)",
        "Max GPU Memory",
    ]

    grouped = grouped[summary_cols]

    grouped["request_type_order"] = grouped["Input Output data Type"].map(
        {name: i for i, name in enumerate(REQUEST_TYPE_ORDER)}
    )
    grouped["scheduler_order"] = grouped["Scheduler"].map(SCHEDULER_ORDER)

    grouped = grouped.sort_values(
        [
            "request_type_order",
            "scheduler_order",
            "Model Name",
            "Number of requests",
            "QPS",
        ],
        na_position="last",
    )

    grouped = grouped.drop(columns=["request_type_order", "scheduler_order"])

    return grouped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmark_output/figure-1"),
    )

    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("log"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("custom_data_output"),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request_level = build_request_level_csv(args.benchmark_root)
    time_logs = parse_time_logs(args.log_root)
    summary = build_summary_csv(request_level, time_logs)

    request_level_output_cols = [
        "Scheduler",
        "Input Output data Type",
        "Model Name",
        "Number of Requests",
        "QPS",
        "Request No",
        "Prefill Token Count",
        "Decode Token Count",
        "Prefill Time",
        "Decode Time",
        "Request Execution Time",
        "Request End-to-End time",
        "Scheduling Delay",
    ]

    request_level_csv = args.output_dir / "custom_request_level_metrics.csv"
    summary_csv = args.output_dir / "custom_summary_metrics.csv"

    request_level[request_level_output_cols].to_csv(request_level_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"Wrote request-level CSV: {request_level_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Request rows: {len(request_level)}")
    print(f"Summary rows: {len(summary)}")

    missing_time = summary[summary["Inference Time"].isna()]
    if len(missing_time) > 0:
        print(f"WARN: {len(missing_time)} summary rows are missing Inference Time.")
        print("This means request-level benchmark data exists, but no matching time-log entry was found.")


if __name__ == "__main__":
    main()