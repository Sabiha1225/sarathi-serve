#!/usr/bin/env python
"""
Generates the same set of grouped-bar charts as the interactive artifact,
as static PNG files: one PNG per (dataset x metric) combination, each a
5-panel grid (one panel per model) with GPU-utilization groups on the x-axis
and the 4 chunk/offload combinations as grouped bars.

Prerequisite (not installed by default in the vllm-018 env):
    pip install matplotlib

Usage:
    python vllm-sarathi-bench/plot_gpu_sweep_summary.py \
        --current /home/sabiha/sarathi-serve/vllm-sarathi-bench/sweep_summary.csv \
        --prefix-caching /home/sabiha/sarathi-serve/vllm-sarathi-bench/sweep_summary_prefix_caching.csv \
        --out-dir /home/sabiha/sarathi-serve/vllm-sarathi-bench/plots
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

MODELS = [
    ("llama2_7b", "Llama-2-7b"),
    ("mistral_7b", "Mistral-7B-v0.3"),
    ("mixtral_7b_8expert", "Mixtral-8x7B (shrunk)"),
    ("qwen_7b", "Qwen-7B"),
    ("yi_6b", "Yi-6B"),
]

GPU_LEVELS = [0.25, 0.45, 0.65, 0.85]

COMBOS = [
    (False, False, "chunk off / offload off", "#2a78d6"),
    (False, True, "chunk off / offload on", "#eb6834"),
    (True, False, "chunk on / offload off", "#1baf7a"),
    (True, True, "chunk on / offload on", "#eda100"),
]

METRICS = [
    ("end_to_end_s", "End-to-end time (s)", lambda v: v),
    ("num_preemptions", "Preemptions (count)", lambda v: v),
    ("kv_offload_gpu_to_cpu_mb", "KV offload GPU\u2192CPU (GB)", lambda v: v / 1000),
    ("kv_offload_cpu_to_gpu_mb", "KV offload CPU\u2192GPU (GB)", lambda v: v / 1000),
    ("tokens_served_from_offload_cache", "Tokens served from offload cache", lambda v: v),
]


def load_rows(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            def num(x):
                return None if x in (None, "") else float(x)

            rows.append({
                "model": row["model"],
                "gpu": num(row["gpu_memory_utilization"]),
                "chunk": row["chunked_prefill"] == "True",
                "offload": row["kv_offloading"] == "True",
                "status": row["status"],
                "end_to_end_s": num(row["end_to_end_s"]),
                "num_preemptions": num(row["num_preemptions"]),
                "kv_offload_gpu_to_cpu_mb": num(row["kv_offload_gpu_to_cpu_mb"]),
                "kv_offload_cpu_to_gpu_mb": num(row["kv_offload_cpu_to_gpu_mb"]),
                "tokens_served_from_offload_cache": num(row["tokens_served_from_offload_cache"]),
            })
    return rows


def find_row(rows, model, gpu, chunk, offload):
    for r in rows:
        if r["model"] == model and r["gpu"] == gpu and r["chunk"] == chunk and r["offload"] == offload:
            return r
    return None


def plot_metric(rows, metric_key, metric_label, transform, dataset_label, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    group_width = 0.8
    bar_width = group_width / len(COMBOS)
    x_base = list(range(len(GPU_LEVELS)))

    for ax_idx, (model_key, model_label) in enumerate(MODELS):
        ax = axes[ax_idx]
        for combo_idx, (chunk, offload, combo_label, color) in enumerate(COMBOS):
            xs, ys = [], []
            missing_xs = []
            for gi, gpu in enumerate(GPU_LEVELS):
                row = find_row(rows, model_key, gpu, chunk, offload)
                x = gi - group_width / 2 + combo_idx * bar_width + bar_width / 2
                val = row[metric_key] if row else None
                if val is None:
                    missing_xs.append(x)
                    continue
                xs.append(x)
                ys.append(transform(val))
            ax.bar(xs, ys, width=bar_width * 0.9, color=color,
                   label=combo_label if ax_idx == 0 else None)
            for mx in missing_xs:
                ax.plot([mx, mx], [0, 0], marker="x", color="0.6", markersize=4)

        ax.set_title(model_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{int(g * 100)}%" for g in GPU_LEVELS], fontsize=9)
        ax.set_xlabel("GPU memory utilization", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # last subplot unused (5 models in a 2x3 grid) -> use it for the legend
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False)

    fig.suptitle(f"{metric_label} \u2014 {dataset_label}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="sweep_summary.csv")
    parser.add_argument("--prefix-caching", default="sweep_summary_prefix_caching.csv")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    datasets = [
        ("current", "Current (unique prompts)", args.current),
        ("prefix_caching", "Pre-fix (shared prompts)", args.prefix_caching),
    ]

    for dataset_key, dataset_label, path in datasets:
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        rows = load_rows(path)
        for metric_key, metric_label, transform in METRICS:
            out_path = os.path.join(args.out_dir, f"{dataset_key}_{metric_key}.png")
            plot_metric(rows, metric_key, metric_label, transform, dataset_label, out_path)


if __name__ == "__main__":
    main()
