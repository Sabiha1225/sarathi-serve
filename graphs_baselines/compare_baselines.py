#!/usr/bin/env python3
"""
Compare vLLM baseline vs sarathi fixed-chunk vs sarathi dynamic-chunk across
all 5 models at a fixed QPS, using kv_block_usage.csv,
kv_block_usage_per_sequence.csv, schedule_iterations.csv, and (sarathi only)
chunk_schedule.csv.

python graphs_baselines/compare_baselines.py --qps 15 --output-dir graphs_baselines/compare_out

"""


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path("/home/sabiha/sarathi-serve/benchmark_output")

CONDITIONS = {
    "vllm": {
        "root": "vllm_baseline",
        "dir_fmt": "arxiv_vllm_poison_{qps}_qps_{model}",
        "color": "#eb6834",
        "label": "vLLM baseline",
        "has_chunking": False,
    },
    "sarathi_fixed": {
        "root": "sarathi_baseline_fixed_chunk",
        "dir_fmt": "arxiv_chunk_512_dyn_false_poison_{qps}_qps_{model}",
        "color": "#2a78d6",
        "label": "sarathi, fixed chunk=512",
        "has_chunking": True,
    },
    "sarathi_dynamic": {
        "root": "sarathi_baseline_dynamic_chunk",
        "dir_fmt": "arxiv_chunk_512_dyn_true_poison_{qps}_qps_{model}",
        "color": "#4a3aa7",
        "label": "sarathi, dynamic chunking",
        "has_chunking": True,
    },
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
})


def run_dir(condition: str, model: str, qps: int) -> Path:
    cfg = CONDITIONS[condition]
    return BENCHMARK_ROOT / cfg["root"] / cfg["dir_fmt"].format(qps=qps, model=model) / "replica_0"


def load_csv(condition: str, model: str, qps: int, filename: str):
    path = run_dir(condition, model, qps) / filename
    if not path.exists():
        print(f"[warn] missing {path}")
        return None
    return pd.read_csv(path)


def prefill_durations(kvp: pd.DataFrame):
    """Iterations from admission to first block-count growth, per sequence."""
    durations = []
    for seq_id, g in kvp.groupby("seq_id"):
        g = g.sort_values("iteration_id")
        first_val = g.iloc[0]["num_blocks_allocated"]
        first_iter = g.iloc[0]["iteration_id"]
        changed = g[g["num_blocks_allocated"] != first_val]
        if len(changed):
            dur = int(changed.iloc[0]["iteration_id"] - first_iter)
        else:
            dur = int(g.iloc[-1]["iteration_id"] - first_iter)
        durations.append(dur)
    return durations


def representative_sequences(kvp: pd.DataFrame, n=6):
    lengths = kvp.groupby("seq_id").size().sort_values(ascending=False)
    candidates = lengths[lengths > 50].index.tolist()
    if not candidates:
        candidates = lengths.index.tolist()
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


# ---------------------------------------------------------------------------
# Chart 1: KV utilization / running sequences / batched tokens, 3-panel timeline
# ---------------------------------------------------------------------------

def chart_timeline(model: str, qps: int, out_dir: Path):
    metrics = [
        ("kv_block_usage.csv", "utilization_pct", "KV utilization, %"),
        ("schedule_iterations.csv", "num_running_sequences", "# running sequences"),
        ("schedule_iterations.csv", "num_batched_tokens", "batched tokens / iter"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (fname, col, ylabel) in zip(axes, metrics):
        for cond, cfg in CONDITIONS.items():
            df = load_csv(cond, model, qps, fname)
            if df is None:
                continue
            ax.plot(df["iteration_id"], df[col], label=cfg["label"], color=cfg["color"], linewidth=1.1)
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("iteration")
    fig.suptitle(f"{model}, QPS={qps} — utilization / concurrency / batch size over time")
    fig.tight_layout()
    out_path = out_dir / f"{model}_qps{qps}_01_timeline.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Chart 2: per-sequence block allocation, 6 representative sequences per condition
# ---------------------------------------------------------------------------

def chart_seq_blocks(model: str, qps: int, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (cond, cfg) in zip(axes, CONDITIONS.items()):
        kvp = load_csv(cond, model, qps, "kv_block_usage_per_sequence.csv")
        if kvp is None:
            ax.set_title(f"{cfg['label']}\n(no data)")
            continue
        seqs = representative_sequences(kvp)
        cmap = plt.get_cmap("tab10")
        for i, seq_id in enumerate(seqs):
            g = kvp[kvp.seq_id == seq_id].sort_values("iteration_id")
            ax.plot(g["iteration_id"], g["num_blocks_allocated"], color=cmap(i % 10), linewidth=1.3, alpha=0.9)
        ax.set_title(cfg["label"], fontsize=10)
        ax.set_xlabel("iteration")
    axes[0].set_ylabel("blocks allocated")
    fig.suptitle(f"{model}, QPS={qps} — per-sequence block allocation (6 representative sequences each)")
    fig.tight_layout()
    out_path = out_dir / f"{model}_qps{qps}_02_seq_blocks.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Chart 3: prefill-plateau duration histogram, overlaid across conditions
# ---------------------------------------------------------------------------

def chart_prefill_hist(model: str, qps: int, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_durations = {}
    for cond, cfg in CONDITIONS.items():
        kvp = load_csv(cond, model, qps, "kv_block_usage_per_sequence.csv")
        if kvp is None:
            continue
        durations = prefill_durations(kvp)
        all_durations[cond] = durations

    if not all_durations:
        print("[warn] no data for prefill histogram")
        return

    max_dur = max(max(d) for d in all_durations.values())
    bins = np.linspace(0, max_dur * 1.02, 25)

    for cond, durations in all_durations.items():
        cfg = CONDITIONS[cond]
        ax.hist(
            durations, bins=bins, histtype="step", linewidth=1.8,
            color=cfg["color"], label=f"{cfg['label']} (median={np.median(durations):.0f})",
        )

    ax.set_xlabel("iterations from admission to first block growth")
    ax.set_ylabel("# sequences")
    ax.legend(fontsize=8)
    fig.suptitle(f"{model}, QPS={qps} — prefill-plateau duration distribution")
    fig.tight_layout()
    out_path = out_dir / f"{model}_qps{qps}_03_prefill_duration_hist.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Chart 4: chunk-size transition scatter, sarathi fixed vs dynamic only (vLLM has no chunking)
# ---------------------------------------------------------------------------

def chart_chunk_scatter(model: str, qps: int, out_dir: Path):
    chunk_conditions = {k: v for k, v in CONDITIONS.items() if v["has_chunking"]}
    fig, axes = plt.subplots(1, len(chunk_conditions), figsize=(11, 4.5), sharey=True)
    if len(chunk_conditions) == 1:
        axes = [axes]

    dataframes = {}
    for cond in chunk_conditions:
        cs = load_csv(cond, model, qps, "chunk_schedule.csv")
        if cs is not None:
            dataframes[cond] = cs

    # Share one explicit color scale across both panels. Without this,
    # matplotlib auto-expands the range for a *constant* array (e.g. the
    # fixed-chunk run, which is always exactly 512) by +/-10%, which makes a
    # perfectly constant chunk size look like it varies. Explicit vmin/vmax
    # also makes color mean the same absolute chunk size in both panels.
    all_sizes = pd.concat([df["chunk_size"] for df in dataframes.values()]) if dataframes else None
    vmin, vmax = (all_sizes.min(), all_sizes.max()) if all_sizes is not None else (0, 1)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1

    sc = None
    for ax, (cond, cfg) in zip(axes, chunk_conditions.items()):
        cs = dataframes.get(cond)
        if cs is None:
            ax.set_title(f"{cfg['label']}\n(no data)")
            continue
        sc = ax.scatter(
            cs["iteration_id"], cs["seq_id"], c=cs["chunk_size"], cmap="viridis",
            vmin=vmin, vmax=vmax, s=6, alpha=0.7,
        )
        ax.set_title(cfg["label"], fontsize=10)
        ax.set_xlabel("iteration")

    if sc is not None:
        fig.colorbar(sc, ax=axes, label="chunk size", fraction=0.046, pad=0.04)

    axes[0].set_ylabel("seq_id (arrival order)")
    fig.suptitle(f"{model}, QPS={qps} — chunk-size transitions (vLLM has no chunking, not shown)")
    fig.tight_layout()
    out_path = out_dir / f"{model}_qps{qps}_04_chunk_scatter.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


MODELS = ["llama", "mistral", "mixtral", "qwen", "yi"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", type=int, default=15, choices=[1, 5, 10, 15, 20])
    parser.add_argument("--output-dir", default="compare_out")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        print(f"\n=== {model}, QPS={args.qps} ===")
        chart_timeline(model, args.qps, out_dir)
        chart_seq_blocks(model, args.qps, out_dir)
        chart_prefill_hist(model, args.qps, out_dir)
        chart_chunk_scatter(model, args.qps, out_dir)

    print(f"\nAll charts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
