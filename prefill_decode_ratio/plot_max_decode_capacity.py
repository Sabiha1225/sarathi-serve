#!/usr/bin/env python3
"""
Plot Experiment 2 (max decode concurrency) results, correctly separating
genuine KV-capacity signal from unrelated crashes.

- max_kv_utilization_pct comes straight from kv_block_usage.csv (trustworthy).
- request_num_restarts (NOT request_num_pauses) is the real preemption
  signal, summed fresh from each run's sequence_metrics.csv.
- "failed" runs are inspected individually: if sequence_metrics.csv exists
  despite the failure (the Kaleido-crash-after-completion case), its data is
  recovered and plotted with a distinct marker; if nothing survived (the
  profiling IndexError / qwen tokenizer bug cases), the point is marked as a
  crash, not a capacity ceiling.

python prefill_decode_ratio/plot_max_decode_capacity.py \
    --summary-csv /home/sabiha/sarathi-serve/benchmark_output/pd_ratio_max_decode_capacity_exp2/max_decode_capacity_summary.csv \
    --output-dir prefill_decode_ratio/exp2_plots
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


def recover_run_metrics(output_dir: str):
    """Best-effort recovery of sequence_metrics.csv / batch_metrics.csv /
    kv_block_usage.csv from a run directory, whether it's marked success or
    failed. Returns a dict with keys: total_restarts, decode_throughput
    (tokens/sec), max_kv_utilization_pct, recovered (bool)."""
    replica_dir = Path(output_dir) / "replica_0"
    seq_csv = replica_dir / "sequence_metrics.csv"
    batch_csv = replica_dir / "batch_metrics.csv"
    kv_csv = replica_dir / "kv_block_usage.csv"

    result = {
        "total_restarts": None,
        "decode_throughput": None,
        "max_kv_utilization_pct": None,
        "recovered": False,
    }

    if seq_csv.exists():
        seq_df = pd.read_csv(seq_csv)
        if "request_num_restarts" in seq_df.columns:
            result["total_restarts"] = int(seq_df["request_num_restarts"].fillna(0).sum())
        result["recovered"] = True

    if batch_csv.exists():
        batch_df = pd.read_csv(batch_csv)
        decode_rows = batch_df[batch_df["batch_num_decode_tokens"] > 0]
        total_decode_tokens = decode_rows["batch_num_decode_tokens"].sum()
        total_decode_time = decode_rows["batch_execution_time"].sum()
        if total_decode_time > 0:
            result["decode_throughput"] = total_decode_tokens / total_decode_time
        result["recovered"] = True

    if kv_csv.exists():
        kv_df = pd.read_csv(kv_csv)
        if len(kv_df):
            result["max_kv_utilization_pct"] = float(kv_df["utilization_pct"].max())
        result["recovered"] = True

    return result


def build_dataset(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df["N"] = df["N"].astype(int)

    restarts, throughputs, kv_util, point_kind = [], [], [], []
    for _, row in df.iterrows():
        rec = recover_run_metrics(row["output_dir"])
        if row["status"] == "success":
            restarts.append(rec["total_restarts"])
            throughputs.append(rec["decode_throughput"])
            kv_util.append(row["max_kv_utilization_pct"])
            point_kind.append("success")
        elif rec["recovered"]:
            restarts.append(rec["total_restarts"])
            throughputs.append(rec["decode_throughput"])
            # kv_block_usage.csv is only recoverable for runs made after the
            # metrics_store.py plot() reordering fix; older crashed runs will
            # have max_kv_utilization_pct=None here even though other files
            # recovered.
            kv_util.append(rec["max_kv_utilization_pct"])
            point_kind.append("recovered_from_crash")
        else:
            restarts.append(None)
            throughputs.append(None)
            kv_util.append(None)
            point_kind.append("no_data_crash")

    df["true_restarts"] = restarts
    df["decode_throughput"] = throughputs
    df["true_kv_utilization_pct"] = kv_util
    df["point_kind"] = point_kind
    return df


def chart_kv_utilization(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, g in df.groupby("model"):
        style = MODEL_STYLE.get(model, {"color": "gray", "label": model})
        clean = g[g["point_kind"] == "success"].sort_values("N")
        ax.plot(
            clean["N"], clean["true_kv_utilization_pct"],
            marker="o", markersize=5, linewidth=1.6,
            color=style["color"], label=style["label"],
        )

        recovered = g[
            (g["point_kind"] == "recovered_from_crash")
            & g["true_kv_utilization_pct"].notna()
        ].sort_values("N")
        if len(recovered):
            ax.scatter(
                recovered["N"], recovered["true_kv_utilization_pct"],
                marker="D", s=50, facecolors="none",
                edgecolors=style["color"], linewidths=1.6, zorder=5,
            )
            if len(clean):
                last_clean = clean.iloc[-1]
                for _, rec_row in recovered.iterrows():
                    ax.plot(
                        [last_clean["N"], rec_row["N"]],
                        [last_clean["true_kv_utilization_pct"], rec_row["true_kv_utilization_pct"]],
                        color=style["color"], linewidth=1.2, linestyle="--", alpha=0.7,
                    )

        crashed = g[
            (g["point_kind"] == "no_data_crash")
            | ((g["point_kind"] == "recovered_from_crash") & g["true_kv_utilization_pct"].isna())
        ].sort_values("N")
        if len(crashed):
            first_n = crashed["N"].min()
            ax.axvline(first_n, color=style["color"], linestyle=":", alpha=0.4, linewidth=1)
            ax.scatter([first_n], [2], marker="x", color=style["color"], s=70, zorder=5)

    ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (concurrent decode sequences)")
    ax.set_ylabel("max KV-block utilization reached in the run (%)")
    ax.set_title(
        "KV-cache saturation vs concurrency\n"
        "(hollow diamond = recovered from a crashed-but-completed run; "
        "x = crashed before any metrics were written, not a memory ceiling)"
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "01_kv_utilization_vs_N.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def chart_decode_throughput(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, g in df.groupby("model"):
        style = MODEL_STYLE.get(model, {"color": "gray", "label": model})

        clean = g[g["point_kind"] == "success"].sort_values("N")
        ax.plot(
            clean["N"], clean["decode_throughput"],
            marker="o", markersize=5, linewidth=1.6,
            color=style["color"], label=style["label"],
        )

        recovered = g[g["point_kind"] == "recovered_from_crash"].sort_values("N")
        if len(recovered):
            ax.scatter(
                recovered["N"], recovered["decode_throughput"],
                marker="D", s=50, facecolors="none",
                edgecolors=style["color"], linewidths=1.6, zorder=5,
            )
            # connect the last clean point to the recovered point so the eye
            # can follow the trend even though the run technically "failed"
            if len(clean):
                last_clean = clean.iloc[-1]
                for _, rec_row in recovered.iterrows():
                    ax.plot(
                        [last_clean["N"], rec_row["N"]],
                        [last_clean["decode_throughput"], rec_row["decode_throughput"]],
                        color=style["color"], linewidth=1.2, linestyle="--", alpha=0.7,
                    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (concurrent decode sequences)")
    ax.set_ylabel("aggregate decode throughput (tokens/sec)")
    ax.set_title("Decode throughput vs concurrency\n(hollow diamond = generation completed but metrics-export crashed — recovered from raw CSVs)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "02_decode_throughput_vs_N.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def print_summary(df: pd.DataFrame):
    print("\nPer-model status at each N (S=success, R=recovered-from-crash, X=no data/crash):")
    for model, g in df.groupby("model"):
        g = g.sort_values("N")
        codes = []
        for _, row in g.iterrows():
            if row["point_kind"] == "success":
                codes.append(f"N={row['N']}:S")
            elif row["point_kind"] == "recovered_from_crash":
                codes.append(f"N={row['N']}:R")
            else:
                codes.append(f"N={row['N']}:X")
        print(f"  {model:8s}: " + "  ".join(codes))

    print("\nTrue preemption count (request_num_restarts, NOT request_num_pauses) at each successful N:")
    for model, g in df.groupby("model"):
        clean = g[g["point_kind"].isin(["success", "recovered_from_crash"])].sort_values("N")
        for _, row in clean.iterrows():
            r = row["true_restarts"]
            print(f"  {model:8s} N={row['N']:<5d} restarts={r if r is not None else 'NA'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default="/home/sabiha/sarathi-serve/benchmark_output/pd_ratio_max_decode_capacity_exp2/max_decode_capacity_summary.csv",
    )
    parser.add_argument("--output-dir", default="./exp2_plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(Path(args.summary_csv))
    chart_kv_utilization(df, out_dir)
    chart_decode_throughput(df, out_dir)
    print_summary(df)

    print(f"\nAll charts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
