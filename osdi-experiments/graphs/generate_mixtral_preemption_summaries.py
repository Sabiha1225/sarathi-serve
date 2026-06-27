#!/usr/bin/env python3

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_ROOT = Path(
    "/home/sabiha/sarathi-serve/benchmark_output/figure-1"
)

SUMMARY_ROOT = Path(
    "/home/sabiha/sarathi-serve/osdi-experiments/graphs/"
    "mixtral_preemption_summaries"
)

QPS_VALUES = [1, 5, 10, 15, 20]
CHUNK_SIZES = [128, 256, 512, 1024, 2048, 3072, 4096]


def find_request_column(df):
    for column in ["Request Id", "request_id", "seq_id"]:
        if column in df.columns:
            return column
    return None


def get_column_sum(df, column):
    if column not in df.columns:
        return 0
    return df[column].fillna(0).sum()


def build_summary(df, qps, chunk_size, run_dir):
    request_column = find_request_column(df)

    total_rows = len(df)
    distinct_requests = (
        df[request_column].nunique()
        if request_column is not None
        else total_rows
    )

    if "total_preemptions" in df.columns:
        preempted_df = df[df["total_preemptions"].fillna(0) > 0]
    else:
        preempted_df = df.iloc[0:0]

    distinct_preempted_requests = (
        preempted_df[request_column].nunique()
        if request_column is not None
        else len(preempted_df)
    )

    return {
        "qps": qps,
        "chunk_size": chunk_size,
        "run_directory": str(run_dir),
        "total_rows": total_rows,
        "distinct_requests": distinct_requests,
        "distinct_preempted_requests": distinct_preempted_requests,
        "percent_requests_preempted": (
            100.0 * distinct_preempted_requests / distinct_requests
            if distinct_requests
            else 0.0
        ),
        "total_preemptions": get_column_sum(df, "total_preemptions"),
        "prefill_preemptions": get_column_sum(
            df, "prefill_preemptions"
        ),
        "decode_preemptions": get_column_sum(
            df, "decode_preemptions"
        ),
        "total_execution_time_sec": get_column_sum(
            df, "execution_time"
        ),
        "total_preempted_time_sec": get_column_sum(
            df, "total_preempted_time"
        ),
        "total_scheduling_delay_sec": get_column_sum(
            df, "scheduling_delay"
        ),
        "mean_execution_time_sec": (
            df["execution_time"].mean()
            if "execution_time" in df.columns
            else 0
        ),
        "mean_preempted_time_sec": (
            df["total_preempted_time"].mean()
            if "total_preempted_time" in df.columns
            else 0
        ),
        "mean_scheduling_delay_sec": (
            df["scheduling_delay"].mean()
            if "scheduling_delay" in df.columns
            else 0
        ),
    }


def write_text_summary(summary, output_file):
    with output_file.open("w") as file:
        file.write("Preemption Summary\n")
        file.write("==================\n")
        file.write(f"QPS: {summary['qps']}\n")
        file.write(f"Chunk size: {summary['chunk_size']}\n")
        file.write(
            f"Distinct requests: "
            f"{summary['distinct_requests']}\n"
        )
        file.write(
            f"Distinct preempted requests: "
            f"{summary['distinct_preempted_requests']}\n"
        )
        file.write(
            f"Percent requests preempted: "
            f"{summary['percent_requests_preempted']:.2f}%\n"
        )
        file.write(
            f"Total preemptions: "
            f"{summary['total_preemptions']}\n"
        )
        file.write(
            f"Prefill preemptions: "
            f"{summary['prefill_preemptions']}\n"
        )
        file.write(
            f"Decode preemptions: "
            f"{summary['decode_preemptions']}\n"
        )


def plot_preemption_counts(df, output_file, title):
    required = {
        "prefill_preemptions",
        "decode_preemptions",
        "total_preemptions",
    }

    if not required.issubset(df.columns):
        print(f"[warning] Missing preemption columns for {output_file}")
        return

    plot_df = (
        df.sort_values("total_preemptions", ascending=False)
        .reset_index(drop=True)
    )

    x = plot_df.index

    plt.figure(figsize=(18, 6))
    plt.bar(
        x,
        plot_df["prefill_preemptions"],
        label="Prefill preemptions",
    )
    plt.bar(
        x,
        plot_df["decode_preemptions"],
        bottom=plot_df["prefill_preemptions"],
        label="Decode preemptions",
    )

    plt.xlabel("Sequence index")
    plt.ylabel("Preemption count")
    plt.title(title)
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_time_breakdown(df, output_file, title):
    required = {
        "execution_time",
        "total_preempted_time",
        "scheduling_delay",
    }

    if not required.issubset(df.columns):
        print(f"[warning] Missing time columns for {output_file}")
        return

    sort_column = (
        "total_preemptions"
        if "total_preemptions" in df.columns
        else "execution_time"
    )

    plot_df = (
        df.sort_values(sort_column, ascending=False)
        .reset_index(drop=True)
    )

    x = plot_df.index
    execution = plot_df["execution_time"].fillna(0)
    preempted = plot_df["total_preempted_time"].fillna(0)
    scheduling = plot_df["scheduling_delay"].fillna(0)

    plt.figure(figsize=(18, 6))
    plt.bar(x, execution, label="Execution time")
    plt.bar(
        x,
        preempted,
        bottom=execution,
        label="Preempted time",
    )
    plt.bar(
        x,
        scheduling,
        bottom=execution + preempted,
        label="Scheduling delay",
    )

    plt.xlabel("Sequence index")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main():
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    missing_runs = []

    for qps in QPS_VALUES:
        for chunk_size in CHUNK_SIZES:
            run_name = (
                f"arxiv_chunk_{chunk_size}_poison_"
                f"{qps}_qps_mixtral"
            )

            run_dir = OUTPUT_ROOT / run_name
            csv_file = (
                run_dir
                / "replica_0"
                / "preemption_metrics.csv"
            )

            if not csv_file.exists():
                print(f"[missing] {csv_file}")
                missing_runs.append(
                    {
                        "qps": qps,
                        "chunk_size": chunk_size,
                        "expected_file": str(csv_file),
                    }
                )
                continue

            print(
                f"[processing] qps={qps}, "
                f"chunk_size={chunk_size}"
            )

            df = pd.read_csv(csv_file)

            run_output_dir = SUMMARY_ROOT / run_name
            run_output_dir.mkdir(parents=True, exist_ok=True)

            summary = build_summary(
                df,
                qps,
                chunk_size,
                run_dir,
            )
            all_summaries.append(summary)

            pd.DataFrame([summary]).to_csv(
                run_output_dir / "preemption_summary.csv",
                index=False,
            )

            write_text_summary(
                summary,
                run_output_dir / "preemption_summary.txt",
            )

            plot_preemption_counts(
                df,
                run_output_dir / "preemption_counts.png",
                (
                    f"Mixtral Preemptions: "
                    f"QPS={qps}, Chunk={chunk_size}"
                ),
            )

            plot_time_breakdown(
                df,
                run_output_dir / "time_breakdown.png",
                (
                    f"Mixtral Time Breakdown: "
                    f"QPS={qps}, Chunk={chunk_size}"
                ),
            )

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df = summary_df.sort_values(
            ["qps", "chunk_size"]
        )

        summary_df.to_csv(
            SUMMARY_ROOT / "mixtral_all_preemption_summary.csv",
            index=False,
        )

        pivot = summary_df.pivot(
            index="chunk_size",
            columns="qps",
            values="total_preemptions",
        )

        pivot.to_csv(
            SUMMARY_ROOT / "mixtral_preemption_pivot.csv"
        )

        print(
            "\nCombined summary saved to:\n"
            f"{SUMMARY_ROOT / 'mixtral_all_preemption_summary.csv'}"
        )

    if missing_runs:
        pd.DataFrame(missing_runs).to_csv(
            SUMMARY_ROOT / "missing_mixtral_runs.csv",
            index=False,
        )

        print(
            f"\nMissing runs: {len(missing_runs)}. "
            "See missing_mixtral_runs.csv"
        )


if __name__ == "__main__":
    main()