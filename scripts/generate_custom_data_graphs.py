#!/usr/bin/env python3
"""
Regenerate the Observation-1 and Sarathi-vs-vLLM comparison graphs as PNGs,
so they can be dropped into a report/doc.

python generate_report_graphs.py \
  --request-csv custom_data_output/custom_request_level_metrics.csv \
  --summary-csv custom_data_output/custom_summary_metrics.csv \
  --output-dir custom_data_output/graphs
python scripts/generate_custom_data_graphs.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Palette (kept consistent with the two artifact reports)
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    "short input short output": "#2a78d6",       # blue   (control)
    "long input short output": "#1baf7a",        # aqua
    "short input long output": "#eda100",        # yellow
    "long input long output": "#008300",         # green
    "mixed long short input output": "#4a3aa7",  # violet
}
CATEGORY_LABELS = {
    "short input short output": "short / short",
    "long input short output": "long in / short out",
    "short input long output": "short in / long out",
    "long input long output": "long / long",
    "mixed long short input output": "mixed",
}
SCHED_COLORS = {"sarathi": "#2a78d6", "vllm": "#eb6834"}

CATEGORY_ORDER = [
    "short input short output",
    "long input short output",
    "short input long output",
    "long input long output",
    "mixed long short input output",
]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
})


def savefig(fig, out_dir: Path, name: str):
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Observation 1 charts (single scheduler, category comparison)
# ---------------------------------------------------------------------------

def chart_baseline_latency(summary: pd.DataFrame, out_dir: Path, scheduler="sarathi", model="llama2_7b"):
    """Bar chart: isolated (N=1) request latency by category."""
    sub = summary[(summary["Scheduler"] == scheduler)
                  & (summary["Model Name"] == model)
                  & (summary["Number of requests"] == 1)]
    sub = sub.set_index("Input Output data Type").loc[CATEGORY_ORDER]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]
    colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER]
    ax.barh(labels, sub["Request Latency"], color=colors)
    for i, v in enumerate(sub["Request Latency"]):
        ax.text(v * 1.02, i, f"{v:.1f}s", va="center", fontsize=9)
    ax.set_xlabel("Request latency, seconds (1 request, no contention)")
    ax.set_title(f"Isolated baseline — {scheduler}, {model}")
    savefig(fig, out_dir, "01_baseline_latency")


def chart_latency_vs_concurrency(summary: pd.DataFrame, out_dir: Path, scheduler="sarathi", model="llama2_7b", qps=1.0):
    """Line chart: request latency vs N concurrent requests, one line per category."""
    sub = summary[(summary["Scheduler"] == scheduler)
                  & (summary["Model Name"] == model)
                  & (summary["QPS"] == qps)]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cat in CATEGORY_ORDER:
        g = sub[sub["Input Output data Type"] == cat].sort_values("Number of requests")
        ax.plot(g["Number of requests"], g["Request Latency"],
                marker="o", markersize=4, linewidth=2,
                color=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat])
    ax.set_yscale("log")
    ax.set_xlabel("Concurrent requests (N)")
    ax.set_ylabel("Request latency, seconds (log scale)")
    ax.set_title(f"Latency vs. concurrency — {scheduler}, {model}, QPS={qps}")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    savefig(fig, out_dir, "02_latency_vs_concurrency")


def chart_tpot_vs_concurrency(summary: pd.DataFrame, out_dir: Path, scheduler="sarathi", model="llama2_7b", qps=1.0):
    sub = summary[(summary["Scheduler"] == scheduler)
                  & (summary["Model Name"] == model)
                  & (summary["QPS"] == qps)]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cat in CATEGORY_ORDER:
        g = sub[sub["Input Output data Type"] == cat].sort_values("Number of requests")
        ax.plot(g["Number of requests"], g["P95 TPOT"] * 1000,
                marker="o", markersize=4, linewidth=2,
                color=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat])
    ax.set_xlabel("Concurrent requests (N)")
    ax.set_ylabel("P95 TPOT, ms")
    ax.set_title(f"P95 TPOT vs. concurrency — {scheduler}, {model}, QPS={qps}")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    savefig(fig, out_dir, "03_tpot_vs_concurrency")


# ---------------------------------------------------------------------------
# Sarathi vs vLLM comparison charts
# ---------------------------------------------------------------------------

def _sched_line_plot(ax, sub, x_col, y_col, y_scale=None):
    for sched in ["sarathi", "vllm"]:
        g = sub[sub["Scheduler"] == sched].sort_values(x_col)
        ax.plot(g[x_col], g[y_col], marker="o", markersize=4, linewidth=2,
                color=SCHED_COLORS[sched], label=sched)
    if y_scale:
        ax.set_yscale(y_scale)
    ax.legend(fontsize=9)


def chart_ttft_tpot_vs_qps(summary: pd.DataFrame, out_dir: Path,
                           model="llama2_7b", category="long input short output", n_requests=10):
    sub = summary[(summary["Model Name"] == model)
                  & (summary["Number of requests"] == n_requests)
                  & (summary["Input Output data Type"] == category)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _sched_line_plot(axes[0], sub, "QPS", "P95 TTFT")
    axes[0].set_xlabel("QPS")
    axes[0].set_ylabel("P95 TTFT, seconds")
    axes[0].set_title("P95 TTFT vs QPS")

    sub2 = sub.copy()
    sub2["P95 TPOT ms"] = sub2["P95 TPOT"] * 1000
    _sched_line_plot(axes[1], sub2, "QPS", "P95 TPOT ms")
    axes[1].set_xlabel("QPS")
    axes[1].set_ylabel("P95 TPOT, ms")
    axes[1].set_title("P95 TPOT vs QPS")

    fig.suptitle(f"{model}, {CATEGORY_LABELS[category]}, N={n_requests} requests")
    savefig(fig, out_dir, "04_ttft_tpot_vs_qps")


def chart_throughput_vs_qps(summary: pd.DataFrame, out_dir: Path,
                             model="llama2_7b", category="long input short output", n_requests=10):
    sub = summary[(summary["Model Name"] == model)
                  & (summary["Number of requests"] == n_requests)
                  & (summary["Input Output data Type"] == category)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _sched_line_plot(axes[0], sub, "QPS", "Request Throughput (req/s)")
    axes[0].set_xlabel("QPS")
    axes[0].set_ylabel("Request throughput, req/s")
    axes[0].set_title("Request throughput vs QPS")

    _sched_line_plot(axes[1], sub, "QPS", "Token Throughput (tokens/s)")
    axes[1].set_xlabel("QPS")
    axes[1].set_ylabel("Token throughput, tokens/s")
    axes[1].set_title("Token throughput vs QPS")

    fig.suptitle(f"{model}, {CATEGORY_LABELS[category]}, N={n_requests} requests")
    savefig(fig, out_dir, "05_throughput_vs_qps")


def chart_latency_breakdown(requests: pd.DataFrame, out_dir: Path,
                             model="llama2_7b", n_requests=10, qps=1.0):
    """Grouped bars: scheduling delay + prefill time (linear), and decode time (log), by category."""
    sub = requests[(requests["Model Name"] == model)
                   & (requests["Number of Requests"] == n_requests)
                   & (requests["QPS"] == qps)]

    means = (sub.groupby(["Input Output data Type", "Scheduler"])
             [["Scheduling Delay", "Prefill Time", "Decode Time"]].mean())

    cats = CATEGORY_ORDER
    x = range(len(cats))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ttft_component = {
        sched: [means.loc[(c, sched), "Scheduling Delay"] + means.loc[(c, sched), "Prefill Time"] for c in cats]
        for sched in ["sarathi", "vllm"]
    }
    ax = axes[0]
    ax.bar([i - width / 2 for i in x], ttft_component["sarathi"], width, color=SCHED_COLORS["sarathi"], label="sarathi")
    ax.bar([i + width / 2 for i in x], ttft_component["vllm"], width, color=SCHED_COLORS["vllm"], label="vllm")
    ax.set_xticks(list(x))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Scheduling delay + prefill time, s")
    ax.set_title("Time-to-first-token components")
    ax.legend(fontsize=9)

    decode = {
        sched: [means.loc[(c, sched), "Decode Time"] for c in cats]
        for sched in ["sarathi", "vllm"]
    }
    ax = axes[1]
    ax.bar([i - width / 2 for i in x], decode["sarathi"], width, color=SCHED_COLORS["sarathi"], label="sarathi")
    ax.bar([i + width / 2 for i in x], decode["vllm"], width, color=SCHED_COLORS["vllm"], label="vllm")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Decode time, s (log scale)")
    ax.set_title("Decode time")
    ax.legend(fontsize=9)

    fig.suptitle(f"Latency breakdown — {model}, N={n_requests}, QPS={qps}")
    savefig(fig, out_dir, "06_latency_breakdown")


def chart_mixed_cliff(summary: pd.DataFrame, out_dir: Path, model="llama2_7b", qps=1.0):
    """Latency vs N for the mixed long/short category, sarathi vs vllm — the HOL-blocking cliff."""
    sub = summary[(summary["Model Name"] == model)
                  & (summary["QPS"] == qps)
                  & (summary["Input Output data Type"] == "mixed long short input output")]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    _sched_line_plot(ax, sub, "Number of requests", "Request Latency", y_scale="log")
    ax.set_xlabel("Concurrent requests (N)")
    ax.set_ylabel("Request latency, seconds (log scale)")
    ax.set_title(f"Mixed long/short category — {model}, QPS={qps}")
    savefig(fig, out_dir, "07_mixed_category_cliff")


def table_robustness(summary: pd.DataFrame, out_dir: Path,
                      models=("llama2_7b", "mistral_7b", "qwen_7b", "yi_6b", "mixtral_7b_8expert"),
                      category="long input short output", n_requests=10, qps=5.0):
    """Rendered table image: TTFT/TPOT sarathi vs vllm across models."""
    sub = summary[(summary["Number of requests"] == n_requests)
                  & (summary["QPS"] == qps)
                  & (summary["Input Output data Type"] == category)]

    rows = []
    for model in models:
        g = sub[sub["Model Name"] == model]
        s = g[g["Scheduler"] == "sarathi"].iloc[0]
        v = g[g["Scheduler"] == "vllm"].iloc[0]
        rows.append([model, f"{s['P95 TTFT']:.2f}s", f"{v['P95 TTFT']:.2f}s",
                     f"{s['P95 TPOT']*1000:.0f}ms", f"{v['P95 TPOT']*1000:.0f}ms"])

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Model", "sarathi TTFT", "vllm TTFT", "sarathi TPOT", "vllm TPOT"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title(f"Robustness across models — {CATEGORY_LABELS[category]}, N={n_requests}, QPS={qps}", pad=20)
    savefig(fig, out_dir, "08_robustness_table")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-csv", default="custom_data_output/custom_request_level_metrics.csv")
    parser.add_argument("--summary-csv", default="custom_data_output/custom_summary_metrics.csv")
    parser.add_argument("--output-dir", default="custom_data_output/graphs")
    parser.add_argument("--model", default="llama2_7b", help="Representative model for single-model charts")
    parser.add_argument("--scheduler", default="sarathi", help="Scheduler for the Observation-1 charts (single-scheduler view)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requests = pd.read_csv(args.request_csv)
    summary = pd.read_csv(args.summary_csv)

    # Observation 1 — single scheduler, category comparison
    chart_baseline_latency(summary, out_dir, scheduler=args.scheduler, model=args.model)
    chart_latency_vs_concurrency(summary, out_dir, scheduler=args.scheduler, model=args.model)
    chart_tpot_vs_concurrency(summary, out_dir, scheduler=args.scheduler, model=args.model)

    # Sarathi vs vLLM comparison
    chart_ttft_tpot_vs_qps(summary, out_dir, model=args.model)
    chart_throughput_vs_qps(summary, out_dir, model=args.model)
    chart_latency_breakdown(requests, out_dir, model=args.model)
    chart_mixed_cliff(summary, out_dir, model=args.model)
    table_robustness(summary, out_dir)

    print(f"\nAll graphs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
