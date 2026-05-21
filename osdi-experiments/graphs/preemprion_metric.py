import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "/home/sabiha/sarathi-serve/benchmark_output/figure-1/arxiv_chunk_128_poison_1_qps/replica_0/preemption_metrics.csv"

df = pd.read_csv(CSV_FILE)

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
plt.savefig("all_sequences_prefill_decode_preemptions.png", dpi=300)
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
plt.savefig("all_sequences_time_breakdown.png", dpi=300)
plt.show()