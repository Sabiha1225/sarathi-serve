#!/usr/bin/env python3
"""
Analyzes ncu kernel-level profiling CSVs for decode-only vs mixed
chunked-prefill+decode workloads, across all five study models.

Reads:
  prefill_decode_ratio/ncu_decode_only_<model>.csv
  prefill_decode_ratio/ncu_mixed_chunked_<model>.csv

Writes (into prefill_decode_ratio/ncu_graphs/):
  ncu_kernel_summary.csv                       per model x workload x kernel-category stats
  fig_utilization_by_category.png              sm__throughput and mem throughput, decode-only vs mixed, per model
  fig_kernel_mix_share_decode_only.png          share of kernel launches by category, per model (decode-only)
  fig_kernel_mix_share_mixed_chunked.png        share of kernel launches by category, per model (mixed chunked)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("/home/sabiha/sarathi-serve/prefill_decode_ratio")
OUT_DIR = DATA_DIR / "ncu_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["llama", "mistral", "mixtral", "qwen", "yi"]
WORKLOADS = ["decode_only", "mixed_chunked"]

# substring -> category, checked in order (first match wins)
KERNEL_CATEGORIES = [
    ("flash_fwd", "attention"),
    ("fused_moe_kernel", "moe_expert_gemm"),
    ("topkgatingsoftmax", "moe_routing"),
    ("moe_align_block_size", "moe_routing"),
    ("rms_norm_kernel", "norm"),
    ("rotary_embedding_kernel", "rotary"),
    ("silu_and_mul_kernel", "activation"),
    ("gemm", "gemm"),
    ("cutlass", "gemm"),
    ("wmma", "gemm"),
    ("reduce_kernel", "reduce"),
    ("distribution_", "rng"),
    ("vectorized_elementwise", "elementwise"),
    ("unrolled_elementwise", "elementwise"),
    ("elementwise_kernel", "elementwise"),
    ("indexselect", "memory_movement"),
    ("catarray", "memory_movement"),
    ("arange", "memory_movement"),
]


def classify_kernel(name: str) -> str:
    n = name.lower()
    for substr, cat in KERNEL_CATEGORIES:
        if substr in n:
            return cat
    return "other"


def find_header_row(path: Path) -> int:
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith('"ID"'):
                return i
    raise ValueError(f"no ncu CSV header found in {path}")


def load_ncu_csv(path: Path) -> pd.DataFrame:
    header_row = find_header_row(path)
    df = pd.read_csv(path, skiprows=header_row)
    df["Metric Value"] = (
        df["Metric Value"].astype(str).str.replace(",", "", regex=False)
    )
    df["Metric Value"] = pd.to_numeric(df["Metric Value"], errors="coerce")

    # long -> wide: one row per kernel launch, one column per metric
    wide = df.pivot_table(
        index=["ID", "Process ID", "Kernel Name"],
        columns="Metric Name",
        values="Metric Value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide["category"] = wide["Kernel Name"].apply(classify_kernel)
    return wide


rows = []
per_model_category = {}

for model in MODELS:
    per_model_category[model] = {}
    for workload in WORKLOADS:
        path = DATA_DIR / f"ncu_{workload}_{model}.csv"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        wide = load_ncu_csv(path)
        wide["dram_bytes_total"] = wide.get(
            "dram__bytes_read.sum", 0
        ).fillna(0) + wide.get("dram__bytes_write.sum", 0).fillna(0)

        n_launches = len(wide)
        by_cat = wide.groupby("category").agg(
            n_launches=("category", "size"),
            mean_sm_throughput_pct=("sm__throughput.avg.pct_of_peak_sustained_elapsed", "mean"),
            mean_mem_throughput_pct=("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "mean"),
            mean_dram_bytes=("dram_bytes_total", "mean"),
        ).reset_index()
        by_cat["launch_share_pct"] = by_cat["n_launches"] / n_launches * 100
        by_cat["model"] = model
        by_cat["workload"] = workload
        rows.append(by_cat)
        per_model_category[model][workload] = by_cat

        print(f"{model:8s} {workload:14s} n_launches={n_launches:6d} "
              f"overall_sm={wide['sm__throughput.avg.pct_of_peak_sustained_elapsed'].mean():5.1f}% "
              f"overall_mem={wide['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].mean():5.1f}%")

summary = pd.concat(rows, ignore_index=True)
summary = summary[["model", "workload", "category", "n_launches", "launch_share_pct",
                    "mean_sm_throughput_pct", "mean_mem_throughput_pct", "mean_dram_bytes"]]
summary.to_csv(OUT_DIR / "ncu_kernel_summary.csv", index=False)
print(f"\nwrote {OUT_DIR / 'ncu_kernel_summary.csv'}")
print(summary.to_string(index=False))

# ---------------------------------------------------------------- fig 1
# overall (all-kernel-average) sm vs mem throughput, decode-only vs mixed, per model
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(MODELS))
width = 0.35

for ax, metric, title in zip(
    axes,
    ["mean_sm_throughput_pct", "mean_mem_throughput_pct"],
    ["Compute throughput (sm__throughput, %)", "Memory throughput (gpu mem throughput, %)"],
):
    for i, workload in enumerate(WORKLOADS):
        vals = []
        for model in MODELS:
            sub = summary[(summary["model"] == model) & (summary["workload"] == workload)]
            total_launches = sub["n_launches"].sum()
            weighted = (sub[metric] * sub["n_launches"]).sum() / total_launches if total_launches else np.nan
            vals.append(weighted)
        ax.bar(x + (i - 0.5) * width, vals, width, label=workload)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_title(title)
    ax.set_ylabel("%")
    ax.legend()

fig.suptitle("Kernel-weighted average utilization: decode-only vs mixed chunked prefill+decode")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_utilization_by_category.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT_DIR / 'fig_utilization_by_category.png'}")

# ---------------------------------------------------------------- fig 2
# kernel-launch category share, per model, one chart per workload
# (highlights Mixtral's moe_routing / moe_expert_gemm categories, absent elsewhere)
def plot_kernel_mix_share(workload: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    all_cats = sorted(summary["category"].unique())
    bottom = np.zeros(len(MODELS))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_cats)))

    for cat, color in zip(all_cats, colors):
        vals = []
        for model in MODELS:
            sub = summary[(summary["model"] == model) & (summary["workload"] == workload) & (summary["category"] == cat)]
            vals.append(sub["launch_share_pct"].sum())
        ax.bar(MODELS, vals, bottom=bottom, label=cat, color=color)
        bottom += np.array(vals)

    ax.set_ylabel("% of kernel launches")
    ax.set_title(f"Kernel-category mix, {workload} workload")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


for workload in WORKLOADS:
    plot_kernel_mix_share(workload, OUT_DIR / f"fig_kernel_mix_share_{workload}.png")
