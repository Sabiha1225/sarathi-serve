#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# python sched_policy_analysis/generate_bar_graphs.py

SRC = Path("/home/sabiha/sarathi_observation2/sched_policy_analysis/Inference_Results(Observation_2_Sched_Policy).csv")
OUT_DIR = Path("/home/sabiha/sarathi_observation2/sched_policy_analysis/bar_graphs")

MODELS = [
    ("meta-llama/Llama-2-7b-hf", "llama2_7b"),
    ("01-ai/Yi-6B", "yi_6b"),
    ("DiscoResearch/mixtral-7b-8expert", "mixtral_7b_8expert"),
]

DATASETS = ["short_short", "short_long", "long_short", "long_long", "mixed"]

# Tier 1 then Tier 2, matching the ordering used everywhere else in this project
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

# adaptive_io_aging is highlighted since it's the policy under study; everything
# else stays a neutral gray so the one bar that matters doesn't get lost in 10 hues
ACCENT = "#2a78d6"
GRAY = "#c3c2b7"


def load_rows():
    with open(SRC) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_safe in MODELS:
        for dataset in DATASETS:
            subset = {
                r["policy"]: r
                for r in rows
                if r["model_name"] == model_name and r["dataset"] == dataset
            }

            for metric_key, metric_label in METRICS:
                labels, values, colors = [], [], []
                for policy in POLICIES:
                    row = subset.get(policy)
                    if row is None:
                        continue
                    labels.append(policy)
                    values.append(float(row[metric_key]))
                    colors.append(ACCENT if policy == "adaptive_io_aging" else GRAY)

                if not values:
                    print(f"[SKIP] no data for {model_safe}/{dataset}/{metric_key}")
                    continue

                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(labels, values, color=colors)

                ax.set_title(f"{metric_label} — {model_safe} — {dataset}")
                ax.set_ylabel(metric_label)
                ax.set_xlabel("Scheduling Policy")
                plt.xticks(rotation=45, ha="right")

                for bar, v in zip(bars, values):
                    label = f"{v:,.2f}" if v < 1000 else f"{v:,.0f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        label,
                        ha="center", va="bottom", fontsize=8,
                    )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()

                out_path = OUT_DIR / model_safe / dataset
                out_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path / f"{metric_key}.png", dpi=150)
                plt.close(fig)
                print(f"[OK] {out_path / (metric_key + '.png')}")


if __name__ == "__main__":
    main()
