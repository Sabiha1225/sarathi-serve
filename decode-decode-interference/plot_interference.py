#!/usr/bin/env python3
"""
Generates the "Five Ways to Wait" charts as PNGs from the decode-decode
interference CSVs:
  - fig_ladder_tpot.png     TPOT: isolated -> 4 mixed conditions, per model, sarathi vs vllm
  - fig_ladder_ttft.png     same for TTFT
  - fig_common_ttft_sarathi.png / _vllm.png   9 common requests, LS vs SS neighbors, TTFT
  - fig_common_tpot_sarathi.png / _vllm.png   same, TPOT

Reads (all in the same directory):
  decode_decode_single_request_fields.csv
  decode_decode_mixed_request_fields.csv
  decode_decode_mixed_request_poisson_qps2_fields.csv
  decode_decode_mixed_request_short_static_fields.csv
  decode_decode_mixed_request_short_poisson_qps2_fields.csv

python3 decode-decode-interference/plot_interference.py

"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("/home/sabiha/sarathi-serve/decode-decode-interference")
OUT_DIR = DATA_DIR / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["llama", "mistral", "mixtral"]
SCHEDULERS = ["sarathi", "vllm"]
PROBE = (11, 41)  # (prefill_tokens, decode_tokens)

ISO_COLOR = "#2a78d6"
LS_COLOR = "#eb6834"
SS_COLOR = "#1baf7a"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- load
single = pd.read_csv(DATA_DIR / "decode_decode_single_request_fields.csv")
ls_static = pd.read_csv(DATA_DIR / "decode_decode_mixed_request_fields.csv")
ls_poisson = pd.read_csv(DATA_DIR / "decode_decode_mixed_request_poisson_qps2_fields.csv")
ss_static = pd.read_csv(DATA_DIR / "decode_decode_mixed_request_short_static_fields.csv")
ss_poisson = pd.read_csv(DATA_DIR / "decode_decode_mixed_request_short_poisson_qps2_fields.csv")

CONDITIONS = [
    ("isolated", None, ISO_COLOR, 1.0),
    ("LS static", ls_static, LS_COLOR, 1.0),
    ("LS Poisson", ls_poisson, LS_COLOR, 0.5),
    ("SS static", ss_static, SS_COLOR, 1.0),
    ("SS Poisson", ss_poisson, SS_COLOR, 0.5),
]


def isolated_mean(scheduler, model, field):
    sub = single[(single.scheduler == scheduler) & (single.model == model)]
    return sub[field].mean()


def probe_value(df, scheduler, model, field):
    sub = df[
        (df.scheduler == scheduler)
        & (df.model == model)
        & (df.request_num_prefill_tokens == PROBE[0])
        & (df.request_num_decode_tokens == PROBE[1])
    ]
    assert len(sub) == 1, f"expected exactly one probe row, got {len(sub)} for {scheduler}/{model} in this condition"
    return sub.iloc[0][field]


def condition_value(cond_df, scheduler, model, field):
    if cond_df is None:
        return isolated_mean(scheduler, model, field)
    return probe_value(cond_df, scheduler, model, field)


# ---------------------------------------------------------------- ladder charts (TPOT, TTFT)
def plot_ladder(field, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(MODELS))
    n_cond = len(CONDITIONS)
    width = 0.8 / n_cond

    for ax, scheduler in zip(axes, SCHEDULERS):
        for i, (label, cond_df, color, alpha) in enumerate(CONDITIONS):
            vals = [condition_value(cond_df, scheduler, model, field) for model in MODELS]
            ax.bar(x + (i - (n_cond - 1) / 2) * width, vals, width, label=label, color=color, alpha=alpha)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_title(scheduler)
        ax.set_ylabel(f"{field} (s)")

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


plot_ladder("tpot", "TPOT: isolated vs. four mixed conditions", OUT_DIR / "fig_ladder_tpot.png")
plot_ladder("ttft", "TTFT: isolated vs. four mixed conditions", OUT_DIR / "fig_ladder_ttft.png")

# ---------------------------------------------------------------- common 9-request charts
COMMON_PAIRS = [(4, 474), (8, 442), (11, 41), (14, 505), (22, 183), (26, 30), (30, 51), (40, 83), (320, 64)]


def plot_common(field, scheduler, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    x = np.arange(len(COMMON_PAIRS))
    width = 0.36
    labels = [f"{pf}/{dc}" for pf, dc in COMMON_PAIRS]

    for ax, model in zip(axes, MODELS):
        ls_vals, ss_vals = [], []
        for pf, dc in COMMON_PAIRS:
            ls_row = ls_static[
                (ls_static.scheduler == scheduler) & (ls_static.model == model)
                & (ls_static.request_num_prefill_tokens == pf) & (ls_static.request_num_decode_tokens == dc)
            ]
            ss_row = ss_static[
                (ss_static.scheduler == scheduler) & (ss_static.model == model)
                & (ss_static.request_num_prefill_tokens == pf) & (ss_static.request_num_decode_tokens == dc)
            ]
            ls_vals.append(ls_row.iloc[0][field])
            ss_vals.append(ss_row.iloc[0][field])

        ax.bar(x - width / 2, ls_vals, width, label="long-output neighbors", color=LS_COLOR)
        ax.bar(x + width / 2, ss_vals, width, label="all-short neighbors", color=SS_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_title(model)
        ax.set_ylabel(f"{field} (s)")

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Nine common requests, two neighborhoods ({field.upper()}, {scheduler}, static arrival)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


for scheduler in SCHEDULERS:
    plot_common("ttft", scheduler, OUT_DIR / f"fig_common_ttft_{scheduler}.png")
    plot_common("tpot", scheduler, OUT_DIR / f"fig_common_tpot_{scheduler}.png")
