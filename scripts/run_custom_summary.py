#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml


# cd /home/sabiha/sarathi-serve

# python scripts/run_custom_summary.py \
#   --benchmark-root benchmark_output/figure-1 \
#   --log-root log \
#   --output-dir custom_data_output

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def short_model_name(model_name):
    name = model_name.split("/")[-1].lower()

    if "llama-2-7b" in name:
        return "llama2_7b"
    if "mistral-7b" in name:
        return "mistral_7b"
    if "mixtral" in name:
        return "mixtral_8x7b"
    if "qwen" in name:
        return "qwen_7b"
    if "yi" in name:
        return "yi_6b"

    return model_name.split("/")[-1].replace("-", "_")


def parse_custom_run_name(run_name):
    pattern = re.compile(
        r"^custom_(?P<scheduler>sarathi|vllm)_"
        r"(?P<request_type>long_input_long_output|long_input_short_output|short_input_long_output|short_input_short_output)_"
        r"(?P<sweep>fixed_num_requests|fixed_qps)_"
    )

    match = pattern.search(run_name)

    if not match:
        return {
            "scheduler_from_name": "unknown",
            "request_type": "unknown",
            "sweep_type": "unknown",
        }

    sweep = match.group("sweep")

    if sweep == "fixed_num_requests":
        sweep_type = "fixed_num_requests_qps_sweep"
    elif sweep == "fixed_qps":
        sweep_type = "fixed_qps_num_requests_sweep"
    else:
        sweep_type = "unknown"

    return {
        "scheduler_from_name": match.group("scheduler"),
        "request_type": match.group("request_type"),
        "sweep_type": sweep_type,
    }


def scheduler_label(config):
    scheduler = str(config.get("replica_scheduler_provider", "unknown"))

    if scheduler == "sarathi":
        chunk_size = config.get("sarathi_scheduler_chunk_size")
        return f"sarathi_chunk_{chunk_size}"

    return scheduler


def parse_number(value):
    if pd.isna(value):
        return float("nan")

    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if match:
        return float(match.group(0))

    return float("nan")


def summarize_series(series, prefix):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return {}

    return {
        f"{prefix}_mean": values.mean(),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_p50": values.quantile(0.50),
        f"{prefix}_p90": values.quantile(0.90),
        f"{prefix}_p95": values.quantile(0.95),
        f"{prefix}_p99": values.quantile(0.99),
    }


def read_gpu_summary(run_dir):
    gpu_path = run_dir / "nvidia-smi-output.csv"

    if not gpu_path.exists():
        return {
            "max_gpu_memory_used_mib": float("nan"),
            "mean_gpu_memory_used_mib": float("nan"),
            "max_gpu_utilization_pct": float("nan"),
            "mean_gpu_utilization_pct": float("nan"),
        }

    df = pd.read_csv(gpu_path)

    memory_col = None
    util_col = None

    for col in df.columns:
        if "memory.used" in col:
            memory_col = col
        if "utilization.gpu" in col:
            util_col = col

    result = {}

    if memory_col:
        memory = df[memory_col].map(parse_number)
        result["max_gpu_memory_used_mib"] = memory.max()
        result["mean_gpu_memory_used_mib"] = memory.mean()
    else:
        result["max_gpu_memory_used_mib"] = float("nan")
        result["mean_gpu_memory_used_mib"] = float("nan")

    if util_col:
        util = df[util_col].map(parse_number)
        result["max_gpu_utilization_pct"] = util.max()
        result["mean_gpu_utilization_pct"] = util.mean()
    else:
        result["max_gpu_utilization_pct"] = float("nan")
        result["mean_gpu_utilization_pct"] = float("nan")

    return result


def parse_time_logs(log_root):
    if log_root is None or not log_root.exists():
        return pd.DataFrame()

    rows = []

    filename_re = re.compile(
        r"^custom_data_time_(?P<scheduler>sarathi|vllm)_"
        r"(?P<request_type>long_input_long_output|long_input_short_output|short_input_long_output|short_input_short_output)\.txt$"
    )

    meta_re = re.compile(
        r"Custom\s+(?P<short_request_type>[\w-]+)\s+(?P<sweep>[\w_]+):\s+"
        r"model=(?P<model>\S+)\s+num_requests=(?P<num_requests>\d+)\s+qps=(?P<qps>[\d.]+)"
    )

    time_re = re.compile(r"Total time taken:\s*(?P<seconds>[\d.]+)\s*seconds")

    for path in sorted(log_root.glob("custom_data_time_*.txt")):
        file_match = filename_re.match(path.name)

        if not file_match:
            continue

        scheduler = file_match.group("scheduler")
        request_type = file_match.group("request_type")

        current = None

        with open(path, "r", errors="ignore") as f:
            for line in f:
                meta = meta_re.search(line)

                if meta:
                    current = {
                        "log_file": str(path),
                        "scheduler_family": scheduler,
                        "request_type": request_type,
                        "sweep_type_from_log": meta.group("sweep"),
                        "model": short_model_name(meta.group("model")),
                        "num_requests": int(meta.group("num_requests")),
                        "qps": float(meta.group("qps")),
                    }
                    continue

                timing = time_re.search(line)

                if timing and current:
                    row = dict(current)
                    row["total_inference_time_sec"] = float(timing.group("seconds"))
                    rows.append(row)
                    current = None

    return pd.DataFrame(rows)


def find_run_dirs(benchmark_root):
    run_dirs = []

    for run_dir in benchmark_root.iterdir():
        if not run_dir.is_dir():
            continue

        if not run_dir.name.startswith("custom_"):
            continue

        config_path = run_dir / "benchmark_config.yml"
        sequence_path = run_dir / "replica_0" / "sequence_metrics.csv"

        if config_path.exists() and sequence_path.exists():
            run_dirs.append(run_dir)

    return sorted(run_dirs)


def load_one_run(run_dir):
    config = read_yaml(run_dir / "benchmark_config.yml")

    sequence_path = run_dir / "replica_0" / "sequence_metrics.csv"
    sequence = pd.read_csv(sequence_path)

    preemption_path = run_dir / "replica_0" / "preemption_metrics.csv"

    if preemption_path.exists():
        preemption = pd.read_csv(preemption_path)

        keep_cols = [
            "Request Id",
            "total_preemptions",
            "num_pauses",
            "num_restarts",
            "prefill_preemptions",
            "decode_preemptions",
            "total_preemption_time_sec",
            "total_preempted_time",
            "execution_time",
            "execution_time_with_preemptions",
            "scheduling_delay",
            "prefill_exec_with_preemption",
            "decode_exec_with_preemption",
            "ete_time",
        ]

        keep_cols = [col for col in keep_cols if col in preemption.columns]
        sequence = sequence.merge(preemption[keep_cols], on="Request Id", how="left")

    parsed_name = parse_custom_run_name(run_dir.name)

    metadata = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "scheduler": scheduler_label(config),
        "scheduler_family": parsed_name["scheduler_from_name"],
        "request_type": parsed_name["request_type"],
        "sweep_type": parsed_name["sweep_type"],
        "model": short_model_name(str(config.get("model_name", "unknown"))),
        "model_full": str(config.get("model_name", "unknown")),
        "num_requests": int(config.get("synthetic_request_generator_num_requests", len(sequence))),
        "qps": float(config.get("poisson_request_interval_generator_qps", 0.0)),
        "tensor_parallel_degree": config.get("model_tensor_parallel_degree"),
        "gpu_memory_utilization_target": config.get("gpu_memory_utilization"),
        "max_model_len": config.get("model_max_model_len"),
    }

    for key, value in metadata.items():
        sequence[key] = value

    numeric_cols = [
        "request_scheduling_delay",
        "request_e2e_time",
        "request_num_prefill_tokens",
        "request_num_decode_tokens",
        "prefill_exec_with_preemption",
        "decode_exec_with_preemption",
        "execution_time_with_preemptions",
        "total_preemptions",
        "num_pauses",
        "total_preemption_time_sec",
    ]

    for col in numeric_cols:
        if col in sequence.columns:
            sequence[col] = pd.to_numeric(sequence[col], errors="coerce")

    output_tokens = sequence["request_num_decode_tokens"].clip(lower=1)
    output_tokens_minus_first = (sequence["request_num_decode_tokens"] - 1).clip(lower=1)

    sequence["ttft_exec_sec"] = sequence["prefill_exec_with_preemption"]

    sequence["ttft_user_sec"] = (
        sequence["request_scheduling_delay"].fillna(0)
        + sequence["prefill_exec_with_preemption"]
    )

    sequence["tpot_sec"] = sequence["decode_exec_with_preemption"] / output_tokens_minus_first
    sequence["tpot_including_first_output_sec"] = sequence["decode_exec_with_preemption"] / output_tokens

    sequence["e2e_sec"] = sequence["request_e2e_time"]
    sequence["scheduling_delay_sec"] = sequence["request_scheduling_delay"]
    sequence["prefill_tokens"] = sequence["request_num_prefill_tokens"]
    sequence["decode_tokens"] = sequence["request_num_decode_tokens"]

    return sequence, metadata


def summarize_one_run(requests, metadata, gpu_summary):
    row = dict(metadata)
    row.update(gpu_summary)

    row["completed_requests"] = len(requests)
    row["total_prefill_tokens"] = requests["prefill_tokens"].sum()
    row["total_decode_tokens"] = requests["decode_tokens"].sum()
    row["mean_prefill_tokens"] = requests["prefill_tokens"].mean()
    row["mean_decode_tokens"] = requests["decode_tokens"].mean()

    metrics = [
        "ttft_exec_sec",
        "ttft_user_sec",
        "tpot_sec",
        "tpot_including_first_output_sec",
        "e2e_sec",
        "scheduling_delay_sec",
        "execution_time_with_preemptions",
        "prefill_exec_with_preemption",
        "decode_exec_with_preemption",
        "total_preemption_time_sec",
        "total_preemptions",
        "num_pauses",
    ]

    for metric in metrics:
        if metric in requests.columns:
            row.update(summarize_series(requests[metric], metric))

    return row


def merge_total_times(run_summary, time_logs):
    if time_logs.empty:
        run_summary["total_inference_time_sec"] = float("nan")
        return run_summary

    grouped = (
        time_logs
        .groupby(
            ["scheduler_family", "request_type", "model", "num_requests", "qps"],
            as_index=False,
        )
        .agg(total_inference_time_sec=("total_inference_time_sec", "mean"))
    )

    merged = run_summary.merge(
        grouped,
        on=["scheduler_family", "request_type", "model", "num_requests", "qps"],
        how="left",
    )

    return merged


def build_report(benchmark_root, log_root):
    request_frames = []
    summary_rows = []

    for run_dir in find_run_dirs(benchmark_root):
        try:
            requests, metadata = load_one_run(run_dir)
            gpu_summary = read_gpu_summary(run_dir)
            summary = summarize_one_run(requests, metadata, gpu_summary)

            request_frames.append(requests)
            summary_rows.append(summary)

        except Exception as exc:
            print(f"WARN: skipping {run_dir}: {exc}")

    if not summary_rows:
        raise RuntimeError(f"No valid custom benchmark runs found under {benchmark_root}")

    run_summary = pd.DataFrame(summary_rows)
    request_level = pd.concat(request_frames, ignore_index=True)
    time_logs = parse_time_logs(log_root)

    run_summary = merge_total_times(run_summary, time_logs)

    if "total_inference_time_sec" in run_summary.columns:
        run_summary["request_throughput_req_per_sec"] = (
            run_summary["completed_requests"] / run_summary["total_inference_time_sec"]
        )
        run_summary["decode_token_throughput_tok_per_sec"] = (
            run_summary["total_decode_tokens"] / run_summary["total_inference_time_sec"]
        )

    sort_cols = [
        "request_type",
        "model",
        "scheduler_family",
        "num_requests",
        "qps",
    ]

    run_summary = run_summary.sort_values(sort_cols)
    request_level = request_level.sort_values(sort_cols + ["Request Id"])

    return run_summary, request_level, time_logs


def write_excel(output_path, run_summary, request_level, time_logs):
    important_cols = [
        "request_type",
        "scheduler_family",
        "scheduler",
        "model",
        "num_requests",
        "qps",
        "completed_requests",
        "total_inference_time_sec",
        "ttft_user_sec_mean",
        "ttft_user_sec_p50",
        "ttft_user_sec_p95",
        "ttft_user_sec_p99",
        "ttft_exec_sec_mean",
        "ttft_exec_sec_p50",
        "ttft_exec_sec_p95",
        "ttft_exec_sec_p99",
        "tpot_sec_mean",
        "tpot_sec_p50",
        "tpot_sec_p95",
        "tpot_sec_p99",
        "scheduling_delay_sec_mean",
        "scheduling_delay_sec_p50",
        "scheduling_delay_sec_p95",
        "scheduling_delay_sec_p99",
        "e2e_sec_mean",
        "e2e_sec_p50",
        "e2e_sec_p95",
        "e2e_sec_p99",
        "execution_time_with_preemptions_mean",
        "execution_time_with_preemptions_p95",
        "prefill_exec_with_preemption_mean",
        "prefill_exec_with_preemption_p95",
        "decode_exec_with_preemption_mean",
        "decode_exec_with_preemption_p95",
        "total_preemptions_mean",
        "num_pauses_mean",
        "max_gpu_memory_used_mib",
        "mean_gpu_memory_used_mib",
        "max_gpu_utilization_pct",
        "mean_gpu_utilization_pct",
        "request_throughput_req_per_sec",
        "decode_token_throughput_tok_per_sec",
        "run_dir",
    ]

    important_cols = [col for col in important_cols if col in run_summary.columns]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        run_summary[important_cols].to_excel(writer, sheet_name="important_summary", index=False)
        run_summary.to_excel(writer, sheet_name="all_run_metrics", index=False)
        request_level.to_excel(writer, sheet_name="request_level", index=False)

        if not time_logs.empty:
            time_logs.to_excel(writer, sheet_name="parsed_time_logs", index=False)


def line_plot(summary, x, y, output_path, title):
    data = summary.dropna(subset=[x, y])

    if data.empty:
        return

    g = sns.relplot(
        data=data,
        x=x,
        y=y,
        hue="scheduler_family",
        style="model",
        col="request_type",
        row="sweep_type",
        kind="line",
        marker="o",
        height=3.4,
        aspect=1.25,
        # sharex=False,
        # sharey=False,
        facet_kws={"sharex": False, "sharey": False},
    )

    g.set_axis_labels(x.replace("_", " "), y.replace("_", " "))
    g.fig.suptitle(title, y=1.02)
    g.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def combined_ttft_tpot_plot(summary, x, output_path, title):
    cols = [
        "scheduler_family",
        "request_type",
        "sweep_type",
        "model",
        x,
        "ttft_user_sec_p50",
        "ttft_user_sec_p95",
        "tpot_sec_p50",
        "tpot_sec_p95",
    ]

    cols = [col for col in cols if col in summary.columns]
    data = summary[cols].dropna(subset=[x])

    rename_map = {
        "ttft_user_sec_p50": "TTFT p50",
        "ttft_user_sec_p95": "TTFT p95",
        "tpot_sec_p50": "TPOT p50",
        "tpot_sec_p95": "TPOT p95",
    }

    metric_cols = [col for col in rename_map if col in data.columns]

    melted = data.melt(
        id_vars=["scheduler_family", "request_type", "sweep_type", "model", x],
        value_vars=metric_cols,
        var_name="metric",
        value_name="seconds",
    )

    melted["metric"] = melted["metric"].map(rename_map)
    melted = melted.dropna(subset=["seconds"])

    if melted.empty:
        return

    g = sns.relplot(
        data=melted,
        x=x,
        y="seconds",
        hue="scheduler_family",
        style="metric",
        col="request_type",
        row="model",
        kind="line",
        marker="o",
        height=3.2,
        aspect=1.25,
        # sharex=False,
        # sharey=False,
        facet_kws={"sharex": False, "sharey": False},
    )

    g.set_axis_labels(x.replace("_", " "), "seconds")
    g.fig.suptitle(title, y=1.02)
    g.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def box_plot(requests, x, y, output_path, title):
    data = requests.dropna(subset=[x, y])

    if data.empty:
        return

    g = sns.catplot(
        data=data,
        x=x,
        y=y,
        hue="scheduler_family",
        col="request_type",
        row="model",
        kind="box",
        showfliers=False,
        height=3.2,
        aspect=1.25,
        sharex=False,
        sharey=False,
        #facet_kws={"sharex": False, "sharey": False},
    )

    g.set_axis_labels(x.replace("_", " "), y.replace("_", " "))
    g.fig.suptitle(title, y=1.02)
    g.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def heatmap(summary, metric, output_path):
    data = summary.dropna(subset=[metric])

    if data.empty:
        return

    groups = list(data.groupby(["request_type", "scheduler_family", "model"]))

    cols = min(3, len(groups))
    rows = (len(groups) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.8), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, ((request_type, scheduler, model), group) in zip(axes.ravel(), groups):
        pivot = group.pivot_table(
            index="qps",
            columns="num_requests",
            values=metric,
            aggfunc="mean",
        )

        ax.set_visible(True)
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap="viridis", ax=ax)
        ax.set_title(f"{request_type}\n{scheduler} | {model}", fontsize=10)
        ax.set_xlabel("num_requests")
        ax.set_ylabel("qps")

    fig.suptitle(metric, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scatter_contention(requests, output_path):
    needed = ["scheduling_delay_sec", "ttft_user_sec", "tpot_sec"]
    data = requests.dropna(subset=needed)

    if data.empty:
        return

    if len(data) > 10000:
        data = data.sample(10000, random_state=7)

    g = sns.relplot(
        data=data,
        x="scheduling_delay_sec",
        y="ttft_user_sec",
        size="tpot_sec",
        hue="scheduler_family",
        col="request_type",
        row="model",
        alpha=0.55,
        height=3.2,
        aspect=1.25,
        # sharex=False,
        # sharey=False,
        facet_kws={"sharex": False, "sharey": False},
    )

    g.set_axis_labels("scheduling delay sec", "TTFT user sec")
    g.fig.suptitle("TTFT contention: scheduling delay vs first-token latency", y=1.02)
    g.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def create_plots(output_dir, run_summary, request_level):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")

    combined_ttft_tpot_plot(
        run_summary,
        "qps",
        plots_dir / "combined_ttft_tpot_vs_qps.png",
        "TTFT and TPOT variation with QPS",
    )

    combined_ttft_tpot_plot(
        run_summary,
        "num_requests",
        plots_dir / "combined_ttft_tpot_vs_num_requests.png",
        "TTFT and TPOT variation with number of requests",
    )

    line_plot(
        run_summary,
        "qps",
        "ttft_user_sec_p95",
        plots_dir / "ttft_p95_vs_qps.png",
        "TTFT p95 vs QPS",
    )

    line_plot(
        run_summary,
        "qps",
        "tpot_sec_p95",
        plots_dir / "tpot_p95_vs_qps.png",
        "TPOT p95 vs QPS",
    )

    line_plot(
        run_summary,
        "num_requests",
        "ttft_user_sec_p95",
        plots_dir / "ttft_p95_vs_num_requests.png",
        "TTFT p95 vs number of requests",
    )

    line_plot(
        run_summary,
        "num_requests",
        "tpot_sec_p95",
        plots_dir / "tpot_p95_vs_num_requests.png",
        "TPOT p95 vs number of requests",
    )

    line_plot(
        run_summary,
        "qps",
        "scheduling_delay_sec_p95",
        plots_dir / "scheduling_delay_p95_vs_qps.png",
        "Scheduling delay p95 vs QPS",
    )

    line_plot(
        run_summary,
        "qps",
        "max_gpu_memory_used_mib",
        plots_dir / "max_gpu_memory_vs_qps.png",
        "Max GPU memory vs QPS",
    )

    box_plot(
        request_level,
        "qps",
        "ttft_user_sec",
        plots_dir / "ttft_distribution_by_qps.png",
        "Exact request-level TTFT distribution by QPS",
    )

    box_plot(
        request_level,
        "qps",
        "tpot_sec",
        plots_dir / "tpot_distribution_by_qps.png",
        "Exact request-level TPOT distribution by QPS",
    )

    box_plot(
        request_level,
        "num_requests",
        "ttft_user_sec",
        plots_dir / "ttft_distribution_by_num_requests.png",
        "Exact request-level TTFT distribution by number of requests",
    )

    box_plot(
        request_level,
        "num_requests",
        "tpot_sec",
        plots_dir / "tpot_distribution_by_num_requests.png",
        "Exact request-level TPOT distribution by number of requests",
    )

    heatmap(run_summary, "ttft_user_sec_p95", plots_dir / "ttft_p95_heatmap.png")
    heatmap(run_summary, "tpot_sec_p95", plots_dir / "tpot_p95_heatmap.png")
    heatmap(run_summary, "scheduling_delay_sec_p95", plots_dir / "scheduling_delay_p95_heatmap.png")

    scatter_contention(request_level, plots_dir / "ttft_contention_scatter.png")


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
        default=Path("benchmark_output/figure-1/custom_analysis"),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_summary, request_level, time_logs = build_report(
        args.benchmark_root,
        args.log_root,
    )

    run_summary_csv = args.output_dir / "run_summary.csv"
    request_level_csv = args.output_dir / "request_level_metrics.csv"
    excel_path = args.output_dir / "custom_benchmark_summary.xlsx"

    run_summary.to_csv(run_summary_csv, index=False)
    request_level.to_csv(request_level_csv, index=False)

    write_excel(excel_path, run_summary, request_level, time_logs)
    create_plots(args.output_dir, run_summary, request_level)

    print(f"Wrote Excel summary: {excel_path}")
    print(f"Wrote run summary CSV: {run_summary_csv}")
    print(f"Wrote request-level CSV: {request_level_csv}")
    print(f"Wrote plots under: {args.output_dir / 'plots'}")
    print(f"Runs summarized: {len(run_summary)}")
    print(f"Request rows summarized: {len(request_level)}")


if __name__ == "__main__":
    main()