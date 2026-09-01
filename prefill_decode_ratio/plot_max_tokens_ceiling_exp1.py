#!/usr/bin/env python3
"""
Plot Experiment 1 (max-tokens-in-one-shot) results from its summary CSV.

python prefill_decode_ratio/plot_max_tokens_ceiling_exp1.py \
    --summary-csv benchmark_output/pd_ratio_max_tokens_ceiling_exp1/max_tokens_ceiling_summary.csv \
    --output-dir prefill_decode_ratio/exp1_plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_STYLE = {
    "llama":   {"color": "#eb6834", "label": "Llama-2-7B"},
    "mistral": {"color": "#2a78d6", "label": "Mistral-7B"},
    "mixtral": {"color": "#4a3aa7", "label": "Mixtral-8x7B (dummy config)"},
    "qwen":    {"color": "#1f9e6b", "label": "Qwen-7B"},
    "yi":      {"color": "#c2185b", "label": "Yi-6B"},
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
    return df


def chart_execution_time(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, g in df.groupby("model"):
        style = MODEL_STYLE.get(model, {"color": "gray", "label": model})
        ok = g[g["status"] == "success"].sort_values("P")
        bad = g[g["status"] != "success"].sort_values("P")

        ax.plot(
            ok["P"], ok["batch_execution_time_sec"],
            marker="o", markersize=4, linewidth=1.6,
            color=style["color"], label=style["label"],
        )
        if len(bad):
            first_fail_p = bad["P"].min()
            ax.axvline(first_fail_p, color=style["color"], linestyle=":", alpha=0.5, linewidth=1)
            ax.scatter([first_fail_p], [ax.get_ylim()[1] * 0.02], marker="x",
                       color=style["color"], s=60, zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("prefill tokens (P), one request, one iteration")
    ax.set_ylabel("batch_execution_time_sec (log scale)")
    ax.set_title("Single-iteration prefill compute cost vs P")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_path = out_dir / "01_execution_time_vs_P.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def chart_throughput(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, g in df.groupby("model"):
        style = MODEL_STYLE.get(model, {"color": "gray", "label": model})
        ok = g[g["status"] == "success"].sort_values("P").copy()
        ok["throughput"] = ok["P"] / ok["batch_execution_time_sec"]

        ax.plot(
            ok["P"], ok["throughput"],
            marker="o", markersize=4, linewidth=1.6,
            color=style["color"], label=style["label"],
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("prefill tokens (P)")
    ax.set_ylabel("throughput = P / batch_execution_time_sec  (tokens/sec)")
    ax.set_title("Prefill throughput vs P — flattening = compute-saturated")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "02_throughput_vs_P.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def print_scaling_summary(df: pd.DataFrame):
    print("\nScaling factor per doubling of P (batch_execution_time ratio; 2.0 = linear, >2.0 = super-linear):")
    for model, g in df.groupby("model"):
        ok = g[g["status"] == "success"].sort_values("P")
        times = ok["batch_execution_time_sec"].to_numpy()
        ps = ok["P"].to_numpy()
        ratios = []
        for i in range(1, len(times)):
            if ps[i] == 2 * ps[i - 1]:
                ratios.append(times[i] / times[i - 1])
        if ratios:
            print(f"  {model:8s}: " + ", ".join(f"{r:.2f}x" for r in ratios))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default="/home/sabiha/sarathi-serve/benchmark_output/pd_ratio_max_tokens_ceiling_exp1/max_tokens_ceiling_summary.csv",
    )
    parser.add_argument("--output-dir", default="./exp1_plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.summary_csv))
    chart_execution_time(df, out_dir)
    chart_throughput(df, out_dir)
    print_scaling_summary(df)

    print(f"\nAll charts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
