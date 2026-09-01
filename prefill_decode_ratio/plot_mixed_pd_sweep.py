#!/usr/bin/env python3
"""
Plot Experiment 3 (mixed prefill+decode, P x chunk_size grid) results.

python prefill_decode_ratio/plot_mixed_pd_sweep.py \
    --summary-csv /home/sabiha/sarathi-serve/benchmark_output/pd_ratio_mixed_sweep_exp3/mixed_sweep_summary.csv \
    --output-dir prefill_decode_ratio/exp3_plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_ORDER = ["llama", "mistral", "mixtral", "qwen", "yi"]
MODEL_LABEL = {
    "llama": "Llama-2-7B",
    "mistral": "Mistral-7B",
    "mixtral": "Mixtral-8x7B (dummy config)",
    "qwen": "Qwen-7B",
    "yi": "Yi-6B",
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


def load(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df["P"] = df["P"].astype(int)
    # chunk_size is "NA" for vllm rows; keep as string for grouping, add a
    # numeric column (None for vllm) for coloring.
    df["chunk_size_num"] = pd.to_numeric(df["chunk_size"], errors="coerce")
    return df


def chart_per_model_ttft_tpot(df: pd.DataFrame, out_dir: Path):
    chunk_sizes = sorted(df.loc[df["scheduler"] == "sarathi", "chunk_size_num"].dropna().unique())
    cmap = matplotlib.colormaps["viridis"].resampled(len(chunk_sizes))
    chunk_color = {c: cmap(i) for i, c in enumerate(chunk_sizes)}

    for model in MODEL_ORDER:
        g_model = df[df["model"] == model]
        if not len(g_model):
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        vllm = g_model[g_model["scheduler"] == "vllm"].sort_values("P")
        axes[0].plot(vllm["P"], vllm["p99_ttft_sec"], marker="o", markersize=5,
                     linewidth=2.2, color="black", linestyle="--", label="vLLM (no chunking)")
        axes[1].plot(vllm["P"], vllm["p99_tpot_sec"], marker="o", markersize=5,
                     linewidth=2.2, color="black", linestyle="--", label="vLLM (no chunking)")

        for c in chunk_sizes:
            g_c = g_model[(g_model["scheduler"] == "sarathi") & (g_model["chunk_size_num"] == c)].sort_values("P")
            if not len(g_c):
                continue
            axes[0].plot(g_c["P"], g_c["p99_ttft_sec"], marker="o", markersize=4,
                         linewidth=1.4, color=chunk_color[c], label=f"sarathi chunk={int(c)}")
            axes[1].plot(g_c["P"], g_c["p99_tpot_sec"], marker="o", markersize=4,
                         linewidth=1.4, color=chunk_color[c], label=f"sarathi chunk={int(c)}")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("P (prefill tokens per request)")
        axes[0].set_ylabel("p99 TTFT (sec)")
        axes[1].set_ylabel("p99 TPOT (sec)")
        axes[0].set_title("TTFT — cost paid by the prefilled request itself")
        axes[1].set_title("TPOT — cost paid by the background decode sequences")
        axes[1].legend(fontsize=7, loc="upper left", ncol=1)

        N = g_model["N"].iloc[0]
        fig.suptitle(f"{MODEL_LABEL[model]} — N={N} background decode sequences, decode_tokens=256")
        fig.tight_layout()
        out_path = out_dir / f"model_{model}_ttft_tpot_vs_P.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def chart_tradeoff_frontier(df: pd.DataFrame, out_dir: Path, P_focus: int):
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4.2 * len(MODEL_ORDER), 4.5), sharey=False)

    for ax, model in zip(axes, MODEL_ORDER):
        g_model = df[(df["model"] == model) & (df["P"] == P_focus)]
        if not len(g_model):
            ax.set_title(f"{MODEL_LABEL[model]}\n(no data at P={P_focus})")
            continue

        sarathi_rows = g_model[g_model["scheduler"] == "sarathi"].sort_values("chunk_size_num")
        vllm_row = g_model[g_model["scheduler"] == "vllm"]

        xs = sarathi_rows["p99_tpot_sec"].to_numpy()
        ys = sarathi_rows["p99_ttft_sec"].to_numpy()
        chunk_vals = sarathi_rows["chunk_size_num"].to_numpy()

        sc = ax.scatter(xs, ys, c=chunk_vals, cmap="viridis", s=70, zorder=3,
                        norm=matplotlib.colors.LogNorm(vmin=chunk_vals.min(), vmax=chunk_vals.max()))
        ax.plot(xs, ys, color="gray", linewidth=1, alpha=0.5, zorder=2)

        for x, y, c in zip(xs, ys, chunk_vals):
            ax.annotate(f"{int(c)}", (x, y), fontsize=7, textcoords="offset points",
                       xytext=(4, 4), color="dimgray")

        if len(vllm_row):
            ax.scatter(vllm_row["p99_tpot_sec"], vllm_row["p99_ttft_sec"],
                      marker="*", s=220, color="crimson", zorder=4, label="vLLM")
            ax.annotate("vLLM", (vllm_row["p99_tpot_sec"].iloc[0], vllm_row["p99_ttft_sec"].iloc[0]),
                       fontsize=8, color="crimson", textcoords="offset points", xytext=(6, -10))

        ax.set_xlabel("p99 TPOT (sec)\n← better for background decode")
        ax.set_title(MODEL_LABEL[model], fontsize=10)
        if model == MODEL_ORDER[0]:
            ax.set_ylabel("p99 TTFT (sec)\n← better for the prefilled request")

    fig.suptitle(f"TTFT vs TPOT tradeoff at P={P_focus} — numbers on points are chunk_size, star is vLLM (no chunking)")
    fig.tight_layout()
    out_path = out_dir / f"tradeoff_frontier_P{P_focus}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def print_summary(df: pd.DataFrame):
    print("\nBest chunk_size per model at the largest P (lowest p99 TPOT among sarathi runs, "
          "with its TTFT cost vs vLLM shown for context):")
    max_p = df["P"].max()
    for model in MODEL_ORDER:
        g = df[(df["model"] == model) & (df["P"] == max_p) & (df["scheduler"] == "sarathi")]
        vllm = df[(df["model"] == model) & (df["P"] == max_p) & (df["scheduler"] == "vllm")]
        if not len(g):
            continue
        best = g.loc[g["p99_tpot_sec"].idxmin()]
        vllm_ttft = vllm["p99_ttft_sec"].iloc[0] if len(vllm) else float("nan")
        vllm_tpot = vllm["p99_tpot_sec"].iloc[0] if len(vllm) else float("nan")
        print(
            f"  {model:8s} P={max_p:<5d} best_chunk={int(best['chunk_size_num']):<5d} "
            f"p99_tpot={best['p99_tpot_sec']:.4f} (vllm={vllm_tpot:.4f})  "
            f"p99_ttft={best['p99_ttft_sec']:.3f} (vllm={vllm_ttft:.3f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default="/home/sabiha/sarathi-serve/benchmark_output/pd_ratio_mixed_sweep_exp3/mixed_sweep_summary.csv",
    )
    parser.add_argument("--output-dir", default="./exp3_plots")
    parser.add_argument("--tradeoff-p", type=int, default=None,
                        help="P value to use for the tradeoff frontier chart (default: max P in the data)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.summary_csv))
    chart_per_model_ttft_tpot(df, out_dir)

    p_focus = args.tradeoff_p or int(df["P"].max())
    chart_tradeoff_frontier(df, out_dir, p_focus)

    print_summary(df)
    print(f"\nAll charts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
