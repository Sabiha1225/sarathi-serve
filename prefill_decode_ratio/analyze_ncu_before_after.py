#!/usr/bin/env python3
"""
Builds decode-only vs mixed-chunked before/after tables and charts for
specific kernel categories from ncu_kernel_summary.csv — used to ground
observation 3 (GEMM: memory-bound -> compute-bound) and observation 4
(norm/rotary: launch-overhead amortization).

Reads:  prefill_decode_ratio/ncu_graphs/ncu_kernel_summary.csv
Writes: prefill_decode_ratio/ncu_graphs/obs3_gemm_table.csv
        prefill_decode_ratio/ncu_graphs/obs4_norm_rotary_table.csv
        prefill_decode_ratio/ncu_graphs/fig_obs3_gemm.png
        prefill_decode_ratio/ncu_graphs/fig_obs4_norm_rotary.png

python3 prefill_decode_ratio/analyze_ncu_before_after.py

"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/sabiha/sarathi-serve/prefill_decode_ratio/ncu_graphs")
summary = pd.read_csv(OUT_DIR / "ncu_kernel_summary.csv")
MODELS = ["llama", "mistral", "mixtral", "qwen", "yi"]


def before_after_table(categories, label):
    """One row per (model, category): decode vs mixed sm/mem throughput + deltas."""
    sub = summary[summary["category"].isin(categories)]
    piv = sub.pivot_table(
        index=["model", "category"],
        columns="workload",
        values=["mean_sm_throughput_pct", "mean_mem_throughput_pct"],
    )
    # flatten the MultiIndex columns: ('mean_sm_throughput_pct','decode_only') -> 'sm_decode_only'
    piv.columns = [f"{metric.replace('mean_', '').replace('_pct', '')}_{wl}" for metric, wl in piv.columns]
    piv = piv.reset_index()
    piv["delta_sm_throughput"] = piv["sm_throughput_mixed_chunked"] - piv["sm_throughput_decode_only"]
    piv["delta_mem_throughput"] = piv["mem_throughput_mixed_chunked"] - piv["mem_throughput_decode_only"]
    piv = piv.sort_values(["category", "model"])
    print(f"\n=== {label} ===")
    print(piv.round(1).to_string(index=False))
    return piv


def plot_before_after(table, categories, title, out_path):
    fig, axes = plt.subplots(
        1, len(categories), figsize=(max(6 * len(categories), 9), 4.5), squeeze=False
    )
    axes = axes[0]
    x = np.arange(len(MODELS))
    width = 0.2

    for ax, cat in zip(axes, categories):
        sub = table[table["category"] == cat].set_index("model").reindex(MODELS)
        ax.bar(x - 1.5 * width, sub["sm_throughput_decode_only"], width, label="sm % (decode)", color="#9DC3D6")
        ax.bar(x - 0.5 * width, sub["sm_throughput_mixed_chunked"], width, label="sm % (mixed)", color="#2F6E8C")
        ax.bar(x + 0.5 * width, sub["mem_throughput_decode_only"], width, label="mem % (decode)", color="#E5C08E")
        ax.bar(x + 1.5 * width, sub["mem_throughput_mixed_chunked"], width, label="mem % (mixed)", color="#C2751E")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_ylabel("% of peak")
        ax.set_title(cat)
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=11, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def moe_before_after_table():
    """launch share + sm/mem throughput, decode vs mixed, mixtral's MoE categories only."""
    sub = summary[(summary["model"] == "mixtral") & (summary["category"].isin(["moe_expert_gemm", "moe_routing"]))]
    piv = sub.pivot_table(
        index=["category"],
        columns="workload",
        values=["launch_share_pct", "mean_sm_throughput_pct", "mean_mem_throughput_pct"],
    )
    piv.columns = [f"{metric.replace('mean_', '').replace('_pct', '')}_{wl}" for metric, wl in piv.columns]
    piv = piv.reset_index()
    for metric in ["launch_share", "sm_throughput", "mem_throughput"]:
        piv[f"delta_{metric}"] = piv[f"{metric}_mixed_chunked"] - piv[f"{metric}_decode_only"]
    print("\n=== Observation 5 — Mixtral MoE kernels, decode-only vs mixed-chunked ===")
    print(piv.round(2).to_string(index=False))
    return piv

def plot_moe_before_after(table, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cats = table["category"].tolist()
    x = np.arange(len(cats))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width / 2, table["launch_share_decode_only"], width, label="decode-only", color="#9DC3D6")
    ax.bar(x + width / 2, table["launch_share_mixed_chunked"], width, label="mixed-chunked", color="#C2751E")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("% of kernel launches")
    ax.set_title("Launch share")
    ax.legend(fontsize=8)

    ax = axes[1]
    w = 0.2
    ax.bar(x - 1.5 * w, table["sm_throughput_decode_only"], w, label="sm % (decode)", color="#9DC3D6")
    ax.bar(x - 0.5 * w, table["sm_throughput_mixed_chunked"], w, label="sm % (mixed)", color="#2F6E8C")
    ax.bar(x + 0.5 * w, table["mem_throughput_decode_only"], w, label="mem % (decode)", color="#E5C08E")
    ax.bar(x + 1.5 * w, table["mem_throughput_mixed_chunked"], w, label="mem % (mixed)", color="#C2751E")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("% of peak")
    ax.set_title("Utilization")
    ax.legend(fontsize=7)

    fig.suptitle("Mixtral MoE kernels: share and efficiency both grow, decode-only -> mixed-chunked", fontsize=11, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

# ---------------------------------------------------------------- obs 3: gemm
obs3 = before_after_table(["gemm"], "Observation 3 — GEMM, decode-only vs mixed-chunked")
obs3.to_csv(OUT_DIR / "obs3_gemm_table.csv", index=False)
plot_before_after(
    obs3, ["gemm"],
    "GEMM: memory-bound in decode-only, more compute-bound once prefill is mixed in",
    OUT_DIR / "fig_obs3_gemm.png",
)

# ---------------------------------------------------------------- obs 4: norm + rotary
obs4 = before_after_table(["norm", "rotary"], "Observation 4 — norm/rotary, decode-only vs mixed-chunked")
obs4.to_csv(OUT_DIR / "obs4_norm_rotary_table.csv", index=False)
plot_before_after(
    obs4, ["norm", "rotary"],
    "norm/rotary: launch-overhead amortization once prefill tokens are mixed in",
    OUT_DIR / "fig_obs4_norm_rotary.png",
)

# ---------------------------------------------------------------- obs 5: MOE
obs5 = moe_before_after_table()
obs5.to_csv(OUT_DIR / "obs5_mixtral_moe_table.csv", index=False)
plot_moe_before_after(obs5, OUT_DIR / "fig_obs5_mixtral_moe.png")