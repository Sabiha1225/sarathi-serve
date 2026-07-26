import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ATTENTION_TYPE = {
    "Mistral_7b": "GQA, 8 KV heads", "mixtral-7b-8expert": "GQA, 8 KV heads", "Yi-6B": "GQA, 4 KV heads",
    "Qwen/Qwen-7B": "MHA, 32 KV heads", "Llama-2-7b-hf": "MHA, 32 KV heads",
}

# python scripts/preemption_gpu_memory_graph.py \
#   "preemption_output/Inference_Results(request_preemption_sarathi).csv" \
#   preemption_output/graphs

def parse_tp_gpu_sweep(path):
    """Two parallel blocks per row: cols 0-8 are the TP-2 run, cols 10-16 are the TP-1 run."""
    raw = pd.read_csv(path, header=None, skiprows=1)
    rows = []
    current_model = None
    current_scheduler = None
    for _, r in raw.iterrows():
        model_cell = str(r[1]) if pd.notna(r[1]) else ""
        if model_cell.strip() and model_cell.strip().lower() != "nan":
            current_model = model_cell.strip()
        label = str(r[4]) if pd.notna(r[4]) else (str(r[12]) if pd.notna(r[12]) else "")
        if "Sarathi Serve" in label:
            current_scheduler = "sarathi"
        elif "vLLM" in label:
            current_scheduler = "vllm"

        def extract(gpu_col, time_col, preempt_col, mem_col, tp_label):
            gpu_cell = str(r[gpu_col]) if pd.notna(r[gpu_col]) else ""
            m = re.search(r"gpu\s*=\s*([\d.]+)", gpu_cell)
            if not m or current_scheduler is None:
                return None
            time_cell = str(r[time_col]).strip() if pd.notna(r[time_col]) else ""
            try:
                inference_time, oom = float(time_cell.replace("s", "").strip()), False
            except ValueError:
                inference_time, oom = None, True
            preempt_cell = str(r[preempt_col]) if pd.notna(r[preempt_col]) else ""
            pm = re.search(r"(\d+)\s*Preemption", preempt_cell)
            mem = r[mem_col] if pd.notna(r[mem_col]) else None
            return dict(model=current_model, scheduler=current_scheduler, tp_config=tp_label,
                        gpu_mem_util=float(m.group(1)), inference_time_s=inference_time, oom=oom,
                        # None (not 0) on OOM — the run never produced a preemption count, it just failed
                        preemption_count=(int(pm.group(1)) if pm else (None if oom else 0)),
                        gpu_memory_mib=float(mem) if mem is not None else None)

        rec = extract(6, 5, 7, 8, "TP-2,PP-1")
        if rec: rows.append(rec)
        rec = extract(14, 13, 15, 16, "TP-1,PP-1")
        if rec: rows.append(rec)
    return pd.DataFrame(rows)


def chart_tp_comparison_per_model(df, out_path):
    models = list(ATTENTION_TYPE.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4.4))
    styles = {("sarathi", "TP-2,PP-1"): ("-", "o"), ("sarathi", "TP-1,PP-1"): ("--", "o"),
              ("vllm", "TP-2,PP-1"): ("-", "s"), ("vllm", "TP-1,PP-1"): ("--", "s")}
    tp_marker = {"TP-2,PP-1": "o", "TP-1,PP-1": "^"}   # circle = 2 GPUs, triangle = 1 GPU
    color_map = {"sarathi": "#2a78d6", "vllm": "#eb6834"}
    for ax, model in zip(axes, models):
        sub = df[df.model == model]
        for (sched, tp), g in sub.groupby(["scheduler", "tp_config"]):
            g = g.sort_values("gpu_mem_util")
            # ls, marker = styles[(sched, tp)]
            ls = "-" if tp == "TP-2,PP-1" else "--"
            marker = tp_marker[tp]
            ax.plot(g.gpu_mem_util, g.inference_time_s, linestyle=ls, marker=marker, markersize=6,
                     color=color_map[sched], label=f"{sched} ({tp})")
            oom = g[g.oom]
            if not oom.empty:
                ylim = ax.get_ylim()
                ax.scatter(oom.gpu_mem_util, [ylim[1] * 0.97] * len(oom), marker="x", s=70,
                           color="red", zorder=5)
        ax.invert_xaxis()
        ax.set_xlabel("gpu_memory_utilization")
        ax.set_title(f"{model}\n({ATTENTION_TYPE[model]})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel("Inference time, s (200 requests, chunk=1024)")
    fig.suptitle("TP-2 vs TP-1: inference time vs GPU memory budget (red x = OOM)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_preemption_vs_gpu_mem(df, out_path):
    models = list(ATTENTION_TYPE.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4.0), sharey=True)
    styles = {("sarathi", "TP-2,PP-1"): ("-", "o"), ("sarathi", "TP-1,PP-1"): ("--", "o"),
              ("vllm", "TP-2,PP-1"): ("-", "s"), ("vllm", "TP-1,PP-1"): ("--", "s")}
    tp_marker = {"TP-2,PP-1": "o", "TP-1,PP-1": "^"}   # circle = 2 GPUs, triangle = 1 GPU
    color_map = {"sarathi": "#2a78d6", "vllm": "#eb6834"}
    for ax, model in zip(axes, models):
        sub = df[df.model == model]
        for (sched, tp), g in sub.groupby(["scheduler", "tp_config"]):
            g = g.sort_values("gpu_mem_util")
            # ls, marker = styles[(sched, tp)]
            ls = "-" if tp == "TP-2,PP-1" else "--"
            marker = tp_marker[tp]
            # NaN preemption_count (OOM rows) breaks the line instead of falsely drawing 0
            ax.plot(g.gpu_mem_util, g.preemption_count, linestyle=ls, marker=marker, markersize=6,
                     color=color_map[sched], label=f"{sched} ({tp})")
        ax.invert_xaxis()
        ax.set_xlabel("gpu_memory_utilization")
        ax.set_title(f"{model}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel("Preemption count (200 requests)")
    fig.suptitle("Preemptions vs GPU memory budget: TP-2 vs TP-1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_tp_slowdown_ratio(df, out_path, gpu_mem_util=0.85):
    """The summary chart: how much you pay for 1 GPU instead of 2, per model, at the healthiest memory setting."""
    models = list(ATTENTION_TYPE.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(models))
    width = 0.35
    for i, sched in enumerate(["sarathi", "vllm"]):
        ratios = []
        for model in models:
            sub = df[(df.model == model) & (df.scheduler == sched) & (df.gpu_mem_util == gpu_mem_util)]
            tp2 = sub[sub.tp_config == "TP-2,PP-1"].inference_time_s
            tp1 = sub[sub.tp_config == "TP-1,PP-1"].inference_time_s
            if len(tp2) and len(tp1) and pd.notna(tp1.iloc[0]) and pd.notna(tp2.iloc[0]):
                ratios.append(tp1.iloc[0] / tp2.iloc[0])
            else:
                ratios.append(float("nan"))
        offset = (i - 0.5) * width
        color = "#2a78d6" if sched == "sarathi" else "#eb6834"
        ax.bar([xi + offset for xi in x], ratios, width, label=sched, color=color)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Inference time ratio: TP-1 / TP-2 (higher = worse penalty for 1 GPU)")
    ax.set_title(f"Cost of dropping from 2 GPUs to 1 GPU, at gpu_memory_utilization={gpu_mem_util}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Inference_Results(request_preemption_sarathi).csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    df = parse_tp_gpu_sweep(path)
    chart_tp_comparison_per_model(df, f"{out_dir}/tp_comparison.png")
    chart_preemption_vs_gpu_mem(df, f"{out_dir}/tp_preemption.png")
    chart_tp_slowdown_ratio(df, f"{out_dir}/tp_slowdown_ratio.png")
    print("done")
