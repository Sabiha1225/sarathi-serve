import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# python scripts/varried_chunk_gpu_sweep_graph_sharegpt.py


SRC = "/home/sabiha/sarathi-serve/varried_chunk/Inference_Results_Sarathi_Serve_ShareGPT_GPU_Sweep.csv"
OUT_DIR = Path("/home/sabiha/sarathi-serve/varried_chunk/sharegpt")

ATTENTION_TYPE = {
    "Mistral_7b": "GQA, 8 KV heads",
    "mixtral-7b-8expert": "GQA, 8 KV heads",
    "Yi-6B": "GQA, 4 KV heads",
    "Qwen/Qwen-7B": "MHA, 32 KV heads",
    "Llama-2-7b-hf": "MHA, 32 KV heads",
}


def parse_gpu_sweep(path):
    """One block per model: col1=model (only on the block's first row),
    col4 marks the scheduler sub-block ("Sarathi Serve..." / "vLLM..."),
    col5=inference time (or "memory error" -> OOM), col6="gpu = X" budget,
    col7="N Preemption" (blank on OOM), col8=peak GPU memory used, MiB."""
    raw = pd.read_csv(path, header=None)
    rows = []
    current_model = None
    current_scheduler = None

    for _, r in raw.iterrows():
        model_cell = str(r[1]) if pd.notna(r[1]) else ""
        if model_cell.strip() and model_cell.strip().lower() != "nan":
            current_model = model_cell.strip()

        label = str(r[4]) if pd.notna(r[4]) else ""
        if "Sarathi Serve" in label:
            current_scheduler = "sarathi"
        elif "vLLM" in label:
            current_scheduler = "vllm"

        gpu_cell = str(r[6]) if pd.notna(r[6]) else ""
        m = re.search(r"gpu\s*=\s*([\d.]+)", gpu_cell)
        if not m or current_scheduler is None or current_model is None:
            continue

        time_cell = str(r[5]).strip() if pd.notna(r[5]) else ""
        try:
            inference_time = float(time_cell.replace("s", "").strip())
            oom = False
        except ValueError:
            inference_time = None
            oom = True

        preempt_cell = str(r[7]) if pd.notna(r[7]) else ""
        pm = re.search(r"(\d+)\s*Preemption", preempt_cell)

        gpu_mem_used = float(r[8]) if pd.notna(r[8]) else None

        rows.append(dict(
            model=current_model,
            scheduler=current_scheduler,
            gpu_mem_util=float(m.group(1)),
            inference_time_s=inference_time,
            oom=oom,
            preemption_count=int(pm.group(1)) if pm else None,
            gpu_mem_used_mib=gpu_mem_used,
        ))

    return pd.DataFrame(rows)


def chart_small_multiples(gdf, value_col, ylabel, title, out_path):
    models = list(gdf.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 4.2), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = gdf[gdf.model == model]
        for sched, style, color in [("sarathi", "-o", "#2a78d6"), ("vllm", "--s", "#eb6834")]:
            g = sub[sub.scheduler == sched].sort_values("gpu_mem_util")
            valid = g.dropna(subset=[value_col])
            ax.plot(valid.gpu_mem_util, valid[value_col], style, label=sched, color=color)

            oom = g[g.oom]
            if not oom.empty:
                ylim = ax.get_ylim()
                y_top = ylim[1] if ylim[1] > 0 else 1
                ax.scatter(oom.gpu_mem_util, [y_top * 0.95] * len(oom),
                           marker="x", s=60, color="red", zorder=5)

        ax.set_xlabel("gpu_memory_utilization")
        ax.set_title(f"{model}\n({ATTENTION_TYPE.get(model, '?')})", fontsize=9)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{title} — ShareGPT, 200 requests, chunk_size=1024 (red x = OOM)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf = parse_gpu_sweep(SRC)

    chart_small_multiples(
        gdf, "inference_time_s", "Inference time, s (200 requests)",
        "Inference Time vs GPU Memory Budget",
        OUT_DIR / "inference_time_vs_gpu_util_sharegpt.png",
    )
    chart_small_multiples(
        gdf, "preemption_count", "Preemption count",
        "Preemption Count vs GPU Memory Budget",
        OUT_DIR / "preemption_count_vs_gpu_util_sharegpt.png",
    )
    chart_small_multiples(
        gdf, "gpu_mem_used_mib", "Peak GPU memory used, MiB",
        "Peak GPU Memory Used vs GPU Memory Budget",
        OUT_DIR / "gpu_memory_used_vs_gpu_util_sharegpt.png",
    )

    print("done")
