import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# python ./scripts/varried_chunk_graph_generation.py "varried_chunk/Inference_Results(Sarathi_Serve).csv" ./varried_chunk

ATTENTION_TYPE = {
    "mistral_7b": "GQA, 8 KV heads",
    "mixtral-7b-8expert": "GQA, 8 KV heads",
    "yi_6b": "GQA, 4 KV heads",
    "Qwen/Qwen-7B": "MHA, 32 KV heads",
    "Llama-2-7b-hf": "MHA, 32 KV heads",
}
IS_GQA = {"mistral_7b": True, "mixtral-7b-8expert": True, "yi_6b": True,
          "Qwen/Qwen-7B": False, "Llama-2-7b-hf": False}
MODEL_COLORS = {
    "mistral_7b": "#2a78d6", "mixtral-7b-8expert": "#1baf7a", "yi_6b": "#4a3aa7",
    "Qwen/Qwen-7B": "#e47200", "Llama-2-7b-hf": "#DC143C",
}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                      "axes.grid": True, "grid.alpha": 0.3})


def parse_sweep(path):
    """Main chunk-size x QPS sweep block (the top part of the sheet)."""
    raw = pd.read_csv(path, header=None, skiprows=1)
    rows = []
    current = {"dataset": None, "model": None, "qps": None}
    for _, r in raw.iterrows():
        model_cell = str(r[1]) if pd.notna(r[1]) else ""
        qps_cell = str(r[6]) if pd.notna(r[6]) else ""
        if model_cell.strip() and model_cell.strip().lower() != "nan":
            current["dataset"] = r[0]
            current["model"] = model_cell.strip()
        m = re.search(r"arrival\s+([\d.]+)\s+requests", qps_cell)
        if m:
            current["qps"] = float(m.group(1))
            continue
        chunk = r[3]
        if pd.notna(chunk) and str(chunk).strip().replace(".", "").isdigit():
            chunk = int(float(chunk))

            def clean_time(v):
                return None if pd.isna(v) else float(str(v).replace("s", "").strip())

            rows.append(dict(dataset=current["dataset"], model=current["model"], qps=current["qps"],
                              chunk_size=chunk, scheduler="sarathi",
                              inference_time_s=clean_time(r[4]), req_throughput=r[7],
                              token_throughput=r[8], output_token_throughput=r[9],
                              ttft_p99=r[10], tpot_p99=r[11], latency_p99=r[12]))
            if pd.notna(r[19]):
                rows.append(dict(dataset=current["dataset"], model=current["model"], qps=current["qps"],
                                  chunk_size=chunk, scheduler="vllm",
                                  inference_time_s=clean_time(r[19]), req_throughput=r[22],
                                  token_throughput=r[23], output_token_throughput=r[24],
                                  ttft_p99=r[25], tpot_p99=r[26], latency_p99=r[27]))
    return pd.DataFrame(rows)


def parse_gpu_preemption_block(path):
    """The chunk_size=1024, 200-request, gpu_memory_utilization sweep near the bottom of each model's section."""
    raw = pd.read_csv(path, header=None, skiprows=1)
    rows = []
    current_model, current_scheduler = None, None
    for _, r in raw.iterrows():
        model_cell = str(r[1]) if pd.notna(r[1]) else ""
        if model_cell.strip() and model_cell.strip().lower() != "nan":
            current_model = model_cell.strip()
        label = str(r[14]) if pd.notna(r[14]) else ""
        if "Sarathi Serve" in label:
            current_scheduler = "sarathi"
        elif "vLLM" in label:
            current_scheduler = "vllm"
        gpu_cell = str(r[16]) if pd.notna(r[16]) else ""
        m = re.search(r"gpu\s*=\s*([\d.]+)", gpu_cell)
        if m and current_scheduler is not None:
            time_cell = str(r[15]).strip() if pd.notna(r[15]) else ""
            try:
                inference_time, oom = float(time_cell.replace("s", "").strip()), False
            except ValueError:
                inference_time, oom = None, True
            preempt_cell = str(r[17]) if pd.notna(r[17]) else ""
            pm = re.search(r"(\d+)\s*Preemption", preempt_cell)
            rows.append(dict(model=current_model, scheduler=current_scheduler,
                              gpu_mem_util=float(m.group(1)), inference_time_s=inference_time,
                              oom=oom, preemption_count=int(pm.group(1)) if pm else None))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

# def chart_inference_time_vs_chunk(df, qps, out_path):
#     """The headline chart: reveals the GQA-vs-MHA split at a given QPS."""
#     sub = df[(df.scheduler == "sarathi") & (df.qps == qps)]
#     vllm_sub = df[(df.scheduler == "vllm") & (df.qps == qps)]
#     fig, ax = plt.subplots(figsize=(7.5, 4.8))
#     for model, g in sub.groupby("model"):
#         g = g.sort_values("chunk_size")
#         style = "-" if IS_GQA.get(model) else "--"
#         ax.plot(g.chunk_size, g.inference_time_s, style, marker="o", markersize=4,
#                 color=MODEL_COLORS.get(model, "gray"),
#                 label=f"{model} ({ATTENTION_TYPE.get(model, '?')})")
#         v = vllm_sub[vllm_sub.model == model]
#         if not v.empty:
#             ax.axhline(v.inference_time_s.iloc[0], color=MODEL_COLORS.get(model, "gray"),
#                        linestyle=":", linewidth=1, alpha=0.6)
#     ax.set_xscale("log", base=2)
#     ax.set_xticks(sorted(sub.chunk_size.unique()))
#     ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
#     ax.set_xlabel("Chunk size")
#     ax.set_ylabel("Inference time, s (1000 requests)")
#     ax.set_title(f"Sarathi inference time vs chunk size — QPS={qps}\n"
#                  f"solid = GQA, dashed = MHA; dotted horizontal = vLLM baseline for that model")
#     ax.legend(fontsize=8)
#     fig.tight_layout()
#     fig.savefig(out_path)
#     plt.close(fig)


# def chart_inference_time_grid(df, out_path):
#     """Small multiples across every QPS level, to confirm the split isn't a fluke of one load level."""
#     qps_values = sorted(df.qps.dropna().unique())
#     fig, axes = plt.subplots(1, len(qps_values), figsize=(4 * len(qps_values), 4), sharey=True)
#     for ax, qps in zip(axes, qps_values):
#         sub = df[(df.scheduler == "sarathi") & (df.qps == qps)]
#         for model, g in sub.groupby("model"):
#             g = g.sort_values("chunk_size")
#             style = "-" if IS_GQA.get(model) else "--"
#             ax.plot(g.chunk_size, g.inference_time_s, style, marker="o", markersize=3,
#                     color=MODEL_COLORS.get(model, "gray"), label=model)
#         ax.set_xscale("log", base=2)
#         ax.set_xticks(sorted(sub.chunk_size.unique()))
#         ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
#         ax.set_title(f"QPS={qps}", fontsize=10)
#         ax.set_xlabel("Chunk size")
#     axes[0].set_ylabel("Inference time, s")
#     axes[-1].legend(fontsize=7, loc="upper right")
#     fig.suptitle("Inference time vs chunk size across load levels (solid=GQA, dashed=MHA)")
#     fig.tight_layout()
#     fig.savefig(out_path)
#     plt.close(fig)


# def chart_ttft_tpot_vs_chunk(df, qps, out_path):
#     """Reproduces the paper's generation-stall trade-off: TTFT vs TPOT as chunk size grows."""
#     sub = df[(df.scheduler == "sarathi") & (df.qps == qps)]
#     fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
#     for model, g in sub.groupby("model"):
#         g = g.sort_values("chunk_size")
#         style = "-" if IS_GQA.get(model) else "--"
#         axes[0].plot(g.chunk_size, g.ttft_p99, style, marker="o", markersize=4,
#                      color=MODEL_COLORS.get(model, "gray"), label=model)
#         axes[1].plot(g.chunk_size, g.tpot_p99, style, marker="o", markersize=4,
#                      color=MODEL_COLORS.get(model, "gray"), label=model)
#     for ax, ylabel, title in zip(axes, ["P99 TTFT, s", "P99 TPOT, s"],
#                                   ["P99 TTFT vs chunk size", "P99 TPOT vs chunk size"]):
#         ax.set_xscale("log", base=2)
#         ax.set_xticks(sorted(sub.chunk_size.unique()))
#         ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
#         ax.set_xlabel("Chunk size")
#         ax.set_ylabel(ylabel)
#         ax.set_title(title)
#         ax.legend(fontsize=7)
#     fig.suptitle(f"QPS={qps}")
#     fig.tight_layout()
#     fig.savefig(out_path)
#     plt.close(fig)

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
        chunk_sizes = sorted(sarathi_sub.chunk_size.unique())
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
    fig.suptitle(f"{title_metric} vs chunk size, by QPS\n"
                 f"solid=GQA dashed=MHA (sarathi); dotted horizontal = vLLM baseline per model")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


METRICS = [
    # (column, y-axis label, title, out-file suffix, log-scale-y)
    ("inference_time_s", "Inference time, s (1000 requests)", "Inference Time", "inference_time", False),
    ("ttft_p99", "P99 TTFT, s", "P99 TTFT", "ttft_p99", False),
    ("tpot_p99", "P99 TPOT, s", "P99 TPOT", "tpot_p99", False),
    ("latency_p99", "P99 Request Latency, s", "Request Latency (P99)", "latency_p99", True),
    ("req_throughput", "Request throughput, req/s", "Request Throughput", "req_throughput", False),
    ("token_throughput", "Token throughput, tokens/s", "Token Throughput", "token_throughput", False),
    ("output_token_throughput", "Output token throughput, tokens/s", "Output Token Throughput",
     "output_token_throughput", False),
]


def chart_gpu_memory_sensitivity(gdf, out_path):
    """Inference time vs GPU memory utilization — shows which models tolerate a smaller KV-cache budget."""
    models = gdf.model.unique()
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 4.2), sharey=False)
    for ax, model in zip(axes, models):
        sub = gdf[gdf.model == model]
        for sched, style in [("sarathi", "-o"), ("vllm", "--s")]:
            g = sub[sub.scheduler == sched].sort_values("gpu_mem_util")
            ax.plot(g.gpu_mem_util, g.inference_time_s, style, label=sched,
                    color="#2a78d6" if sched == "sarathi" else "#eb6834")
            oom = g[g.oom]
            if not oom.empty:
                ax.scatter(oom.gpu_mem_util, [ax.get_ylim()[1] * 0.95] * len(oom),
                           marker="x", s=60, color="red", zorder=5)
        ax.set_xlabel("gpu_memory_utilization")
        ax.set_title(f"{model}\n({ATTENTION_TYPE.get(model, '?')})", fontsize=9)
        ax.invert_xaxis()
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Inference time, s (200 requests, chunk=1024)")
    fig.suptitle("Sensitivity to reduced GPU memory budget (red x = OOM)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Inference_Results(Sarathi_Serve).csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    df = parse_sweep(path)
    gdf = parse_gpu_preemption_block(path)

    for metric, ylabel, title_metric, suffix, log_y in METRICS:
        chart_metric_vs_chunk_by_qps(df, metric, ylabel, title_metric,
                                      out_path=f"{out_dir}/{suffix}_vs_chunk_by_qps.png", log_y=log_y)

    chart_gpu_memory_sensitivity(gdf, out_path=f"{out_dir}/gpu_memory_sensitivity.png")
    print("done")



# if __name__ == "__main__":
#     import sys
#     path = sys.argv[1] if len(sys.argv) > 1 else "Inference_Results(Sarathi_Serve).csv"
#     out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

#     df = parse_sweep(path)
#     gdf = parse_gpu_preemption_block(path)

#     chart_inference_time_vs_chunk(df, qps=1.0, out_path=f"{out_dir}/01_inference_time_vs_chunk_qps1.png")
#     chart_inference_time_grid(df, out_path=f"{out_dir}/02_inference_time_grid_all_qps.png")
#     chart_ttft_tpot_vs_chunk(df, qps=1.0, out_path=f"{out_dir}/03_ttft_tpot_vs_chunk_qps1.png")
#     chart_gpu_memory_sensitivity(gdf, out_path=f"{out_dir}/04_gpu_memory_sensitivity.png")
#     print("done")
