import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# python ./scripts/varried_chunk_graph_generation_sharegpt.py "varried_chunk/Inference_Results(Sarathi_Serve_ShareGPT).csv" ./varried_chunk

ATTENTION_TYPE = {
    "mistral": "GQA, 8 KV heads",
    "mixtral": "GQA, 8 KV heads",
    "yi": "GQA, 4 KV heads",
    "qwen": "MHA, 32 KV heads",
    "llama": "MHA, 32 KV heads",
}
IS_GQA = {"mistral": True, "mixtral": True, "yi": True, "qwen": False, "llama": False}
MODEL_COLORS = {
    "mistral": "#2a78d6", "mixtral": "#1baf7a", "yi": "#4a3aa7",
    "qwen": "#e47200", "llama": "#DC143C",
}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                      "axes.grid": True, "grid.alpha": 0.3})


def parse_sweep(path):
    """This CSV is already tidy: Sarathi block in columns 1-13, vLLM block in
    columns 17-28, one row per (model, chunk_size, qps) for Sarathi and one row
    per (model, qps) for vLLM (chunk_size doesn't apply there)."""
    raw = pd.read_csv(path, header=None, skiprows=1)

    sarathi_cols = ["model", "chunk_size", "qps", "inference_time_s", "ttft_p99", "tpot_p99",
                     "latency_p99", "req_throughput", "token_throughput", "output_token_throughput",
                     "scheduling_delay_p99", "preemption_count", "max_gpu_memory_mib"]
    sarathi = raw.iloc[:, 1:14].copy()
    sarathi.columns = sarathi_cols
    sarathi = sarathi.dropna(subset=["model"])
    sarathi = sarathi[sarathi["model"] != "model"]  # in case a stray header row slipped in
    sarathi["scheduler"] = "sarathi"

    vllm_cols = ["model", "qps", "inference_time_s", "ttft_p99", "tpot_p99",
                 "latency_p99", "req_throughput", "token_throughput", "output_token_throughput",
                 "scheduling_delay_p99", "preemption_count", "max_gpu_memory_mib"]
    vllm = raw.iloc[:, 17:29].copy()
    vllm.columns = vllm_cols
    vllm = vllm.dropna(subset=["model"])
    vllm = vllm[vllm["model"] != "model"]
    vllm["scheduler"] = "vllm"
    vllm["chunk_size"] = None

    df = pd.concat([sarathi, vllm], ignore_index=True)

    numeric_cols = ["chunk_size", "qps", "inference_time_s", "ttft_p99", "tpot_p99", "latency_p99",
                     "req_throughput", "token_throughput", "output_token_throughput",
                     "scheduling_delay_p99", "preemption_count", "max_gpu_memory_mib"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def chart_metric_vs_chunk_by_qps(df, metric, ylabel, title_metric, out_path, log_y=False):
    """One panel per QPS; each panel shows every model's sarathi curve across chunk sizes
    (solid=GQA, dashed=MHA) plus a dotted horizontal line for that model's vLLM value at the same QPS."""
    qps_values = sorted(df.qps.dropna().unique())
    fig, axes = plt.subplots(1, len(qps_values), figsize=(4.2 * len(qps_values), 4.6), sharey=True)
    if len(qps_values) == 1:
        axes = [axes]
    for ax, qps in zip(axes, qps_values):
        sarathi_sub = df[(df.scheduler == "sarathi") & (df.qps == qps)]
        vllm_sub = df[(df.scheduler == "vllm") & (df.qps == qps)]
        chunk_sizes = sorted(sarathi_sub.chunk_size.dropna().unique())
        for model, g in sarathi_sub.groupby("model"):
            g = g.sort_values("chunk_size")
            style = "-" if IS_GQA.get(model) else "--"
            color = MODEL_COLORS.get(model, "gray")
            ax.plot(g.chunk_size, g[metric], style, marker="o", markersize=4, color=color, label=model)
            v = vllm_sub[vllm_sub.model == model]
            if not v.empty and pd.notna(v[metric].iloc[0]):
                ax.axhline(v[metric].iloc[0], color=color, linestyle=":", linewidth=1.3, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(chunk_sizes)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Chunk size")
        ax.set_title(f"QPS={qps}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title_metric} vs chunk size, by QPS — ShareGPT\n"
                 f"solid=GQA dashed=MHA (sarathi); dotted horizontal = vLLM baseline per model")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


METRICS = [
    # (column, y-axis label, title, out-file suffix, log-scale-y)
    ("inference_time_s", "Inference time, s (300 requests)", "Inference Time", "inference_time", False),
    ("ttft_p99", "P99 TTFT, s", "P99 TTFT", "ttft_p99", False),
    ("tpot_p99", "P99 TPOT, s", "P99 TPOT", "tpot_p99", False),
    ("latency_p99", "P99 Request Latency, s", "Request Latency (P99)", "latency_p99", True),
    ("req_throughput", "Request throughput, req/s", "Request Throughput", "req_throughput", False),
    ("token_throughput", "Token throughput, tokens/s", "Token Throughput", "token_throughput", False),
    ("output_token_throughput", "Output token throughput, tokens/s", "Output Token Throughput",
     "output_token_throughput", False),
    ("max_gpu_memory_mib", "Peak GPU memory, MiB", "Peak GPU Memory", "max_gpu_memory", False),
]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Inference_Results(Sarathi_Serve_ShareGPT).csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    df = parse_sweep(path)

    for metric, ylabel, title_metric, suffix, log_y in METRICS:
        chart_metric_vs_chunk_by_qps(df, metric, ylabel, title_metric,
                                      out_path=f"{out_dir}/{suffix}_vs_chunk_by_qps_sharegpt.png", log_y=log_y)

    print("done")
