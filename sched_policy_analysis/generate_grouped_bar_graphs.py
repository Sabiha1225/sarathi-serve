#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# python sched_policy_analysis/generate_grouped_bar_graphs.py

SRC = Path("/home/sabiha/sarathi_observation2/sched_policy_analysis/Inference_Results(Observation_2_Sched_Policy).csv")
OUT_DIR = Path("/home/sabiha/sarathi_observation2/sched_policy_analysis/dataset_grouped_bar_graphs")

MODELS = [
    ("meta-llama/Llama-2-7b-hf", "llama2_7b"),
    ("01-ai/Yi-6B", "yi_6b"),
    ("DiscoResearch/mixtral-7b-8expert", "mixtral_7b_8expert"),
]

DATASETS = ["short_short", "short_long", "long_short", "long_long", "mixed"]

POLICIES = [
    "fcfs", "sjf", "ljf", "prompt_len_sjf", "prefill_aging_fairness", "adaptive_io_aging",
    "vtc_fairness", "skip_join_mlfq", "hybrid", "least_laxity_first",
]

METRICS = [
    ("inference_time", "Inference Time (s)"),
    ("ttft_p99", "TTFT p99 (s)"),
    ("tpot_p99", "TPOT p99 (s)"),
    ("request_latency_p99", "Request Latency p99 (s)"),
    ("scheduling_delay_p99", "Scheduling Delay p99 (s)"),
    ("preemption_count_waiting", "Preemption Count (waiting)"),
]

GRAY = "#c3c2b7"
MIN_COLOR = "#0ca30c"   # status "good" — this policy won this workload
ACCENT = "#2a78d6"      # outline only — marks adaptive_io_aging regardless of fill
ACCENT_POLICY = "adaptive_io_aging"

CLUSTER_GAP = 3  # extra bar-widths of empty space between dataset clusters


def load_rows():
    with open(SRC) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_policies = len(POLICIES)
    cluster_span = n_policies + CLUSTER_GAP

    for model_name, model_safe in MODELS:
        model_rows = [r for r in rows if r["model_name"] == model_name]

        for metric_key, metric_label in METRICS:
            fig, ax = plt.subplots(figsize=(20, 6))

            xs, values, fills, edges, edge_widths = [], [], [], [], []
            tick_positions, tick_labels = [], []
            cluster_centers = []

            for di, dataset in enumerate(DATASETS):
                base = di * cluster_span

                cluster_values = {}
                for policy in POLICIES:
                    row = next(
                        (r for r in model_rows if r["policy"] == policy and r["dataset"] == dataset),
                        None,
                    )
                    cluster_values[policy] = float(row[metric_key]) if row else None

                valid = {p: v for p, v in cluster_values.items() if v is not None}
                min_policy = min(valid, key=valid.get) if valid else None

                for pi, policy in enumerate(POLICIES):
                    x = base + pi
                    v = cluster_values[policy] or 0
                    xs.append(x)
                    values.append(v)
                    fills.append(MIN_COLOR if policy == min_policy else GRAY)
                    if policy == ACCENT_POLICY:
                        edges.append(ACCENT)
                        edge_widths.append(2.2)
                    else:
                        edges.append("none")
                        edge_widths.append(0)
                    tick_positions.append(x)
                    tick_labels.append(policy)

                cluster_centers.append((base + (n_policies - 1) / 2, dataset))

            ax.bar(xs, values, width=0.85, color=fills, edgecolor=edges, linewidth=edge_widths)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)

            for cx, dataset in cluster_centers:
                ax.annotate(
                    dataset,
                    xy=(cx, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -62), textcoords="offset points",
                    ha="center", va="top", fontsize=11, fontweight="bold",
                    annotation_clip=False,
                )

            for di in range(1, len(DATASETS)):
                sep_x = di * cluster_span - (CLUSTER_GAP / 2) - 0.5
                ax.axvline(sep_x, color="#c3c2b7", linewidth=1, linestyle="-", alpha=0.6)

            legend_handles = [
                Patch(facecolor=MIN_COLOR, label="Best (minimum) in this workload"),
                Patch(facecolor=GRAY, label="Other policies"),
                Patch(facecolor=GRAY, edgecolor=ACCENT, linewidth=2.2, label="adaptive_io_aging (ours)"),
            ]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

            ax.set_title(f"{metric_label} — {model_safe} — policies grouped by workload")
            ax.set_ylabel(metric_label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_xlim(-1, len(DATASETS) * cluster_span - CLUSTER_GAP)

            plt.subplots_adjust(bottom=0.28)

            out_path = OUT_DIR / model_safe
            out_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path / f"{metric_key}.png", dpi=150)
            plt.close(fig)
            print(f"[OK] {out_path / (metric_key + '.png')}")


if __name__ == "__main__":
    main()
