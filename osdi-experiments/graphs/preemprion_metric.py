import pandas as pd
import matplotlib.pyplot as plt

# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_128_poison_1_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_256_poison_1_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_512_poison_1_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_1024_poison_1_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_2048_poison_1_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_3072_poison_1_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_4096_poison_1_qps/replica_0/preemption_metrics.csv"

# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_128_poison_25_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_256_poison_25_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_512_poison_25_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_1024_poison_25_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_2048_poison_25_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_3072_poison_25_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_4096_poison_25_qps/replica_0/preemption_metrics.csv"

# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_128_poison_50_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_256_poison_50_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_512_poison_50_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_1024_poison_50_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_2048_poison_50_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_3072_poison_50_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_4096_poison_50_qps/replica_0/preemption_metrics.csv"


# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_128_poison_100_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_256_poison_100_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_512_poison_100_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_1024_poison_100_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_2048_poison_100_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_3072_poison_100_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_4096_poison_100_qps/replica_0/preemption_metrics.csv"


# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_vllm_no_poison/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_vllm_poison_1_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_vllm_poison_25_qps/replica_0/preemption_metrics.csv"
# CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_vllm_poison_50_qps/replica_0/preemption_metrics.csv"
CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_vllm_poison_100_qps/replica_0/preemption_metrics.csv"

df = pd.read_csv(CSV_FILE)

def print_preemption_summary(df):
    request_col = None

    if "Request Id" in df.columns:
        request_col = "Request Id"
    elif "request_id" in df.columns:
        request_col = "request_id"
    elif "seq_id" in df.columns:
        request_col = "seq_id"

    preempted_df = df[df["total_preemptions"] > 0]

    total_rows = len(df)
    distinct_requests = df[request_col].nunique() if request_col else total_rows
    distinct_preempted_requests = (
        preempted_df[request_col].nunique() if request_col else len(preempted_df)
    )

    # percent_preempted = (
    #     100 * distinct_preempted_requests / distinct_requests
    #     if distinct_requests > 0
    #     else 0
    # )

    print("\n================ Preemption Summary ================")
    print(f"Total rows/sequences: {total_rows}")
    print(f"Distinct requests: {distinct_requests}")
    print(f"Distinct preempted requests: {distinct_preempted_requests}")
    # print(f"Percent requests preempted: {percent_preempted:.2f}%")

    print("\n---------------- Preemption Counts ----------------")
    print(f"Total preemptions: {df['total_preemptions'].sum()}")
    print(f"Total prefill preemptions: {df['prefill_preemptions'].sum()}")
    print(f"Total decode preemptions: {df['decode_preemptions'].sum()}")

    # print("\n---------------- Time Summary ----------------")
    # if "total_preempted_time" in df.columns:
    #     print(f"Total preempted time: {df['total_preempted_time'].sum():.4f} sec")

    # if "total_preemption_time_sec" in df.columns:
    #     print(f"Total preemption time: {df['total_preemption_time_sec'].sum():.4f} sec")

    # if "execution_time" in df.columns:
    #     print(f"Total execution time: {df['execution_time'].sum():.4f} sec")

    # if "scheduling_delay" in df.columns:
    #     print(f"Total scheduling delay: {df['scheduling_delay'].sum():.4f} sec")

    # print("\n---------------- Average Per Request ----------------")
    # print(f"Average preemptions per request: {df['total_preemptions'].mean():.4f}")

    # if len(preempted_df) > 0:
    #     print(
    #         f"Average preemptions per preempted request: "
    #         f"{preempted_df['total_preemptions'].mean():.4f}"
    #     )
    # else:
    #     print("Average preemptions per preempted request: 0.0000")

    print("====================================================\n")

# Use sequence/request id if available, otherwise row index
if "seq_id" in df.columns:
    x = df["seq_id"].astype(str)
elif "request_id" in df.columns:
    x = df["request_id"].astype(str)
else:
    x = df.index.astype(str)

# Keep all 1000 sequences
df_plot = df.copy()

# Optional: sort to make the plot easier to understand
# Comment this out if you want original order
df_plot = df_plot.sort_values("total_preemptions", ascending=False).reset_index(drop=True)
x = df_plot.index.astype(str)

# =====================================================
# Graph 1: all 1000 sequences, prefill + decode counts
# =====================================================
plt.figure(figsize=(28, 7))

plt.bar(
    x,
    df_plot["prefill_preemptions"],
    label="Prefill preemptions",
)

plt.bar(
    x,
    df_plot["decode_preemptions"],
    bottom=df_plot["prefill_preemptions"],
    label="Decode preemptions",
)

plt.xlabel("Sequence index")
plt.ylabel("Preemption count")
plt.title("Prefill and Decode Preemptions Across All Sequences")
plt.xticks([])  # hide 1000 labels because they are unreadable
plt.legend()
plt.tight_layout()
plt.savefig("arxiv_vllm_poison_100_qps_prefill_decode_preemptions.png", dpi=300)
plt.show()


# =====================================================
# Graph 2: all 1000 sequences, time breakdown
# =====================================================
plt.figure(figsize=(28, 7))

plt.bar(
    x,
    df_plot["execution_time"],
    label="Execution time",
)

plt.bar(
    x,
    df_plot["total_preempted_time"],
    bottom=df_plot["execution_time"],
    label="Total preempted time",
)

plt.bar(
    x,
    df_plot["scheduling_delay"],
    bottom=df_plot["execution_time"] + df_plot["total_preempted_time"],
    label="Scheduling delay",
)

plt.xlabel("Sequence index")
plt.ylabel("Time (seconds)")
plt.title("Execution, Preemption, and Scheduling Delay Across All Sequences")
plt.xticks([])  # hide 1000 labels because they are unreadable
plt.legend()
plt.tight_layout()
plt.savefig("arxiv_vllm_poison_100_qps_time_breakdown.png", dpi=300)
plt.show()

print_preemption_summary(df)