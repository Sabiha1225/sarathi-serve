import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

s2 = pd.read_csv("custom_summary_metrics.csv").rename(columns=lambda c: c.strip())
s1 = pd.read_csv("custom_summary_metrics_tp1_pp1.csv").rename(columns=lambda c: c.strip())

key = ["Scheduler", "Input Output data Type", "Model Name", "Number of requests", "QPS"]
m = s2.merge(s1, on=key, suffixes=("_tp2", "_tp1"))
m["ttft_ratio"] = m["P95 TTFT_tp1"] / m["P95 TTFT_tp2"]

models = ["llama2_7b", "mistral_7b", "qwen_7b", "yi_6b", "mixtral_7b_8expert"]
g = m.groupby(["Model Name", "Scheduler"])["ttft_ratio"].median().unstack()
g = g.reindex(models)

fig, ax = plt.subplots(figsize=(8, 4.8))
x = range(len(models))
width = 0.35
ax.bar([i - width / 2 for i in x], g["sarathi"], width, label="sarathi", color="#2a78d6")
ax.bar([i + width / 2 for i in x], g["vllm"], width, label="vllm", color="#eb6834")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
ax.text(len(models) - 0.6, 1.05, "TP-1 slower than TP-2", fontsize=8, color="gray")
ax.text(len(models) - 0.6, 0.90, "TP-1 faster than TP-2", fontsize=8, color="gray", ha="right")
ax.set_xticks(list(x))
ax.set_xticklabels(models, rotation=25, ha="right")
ax.set_ylabel("P95 TTFT ratio: TP-1 / TP-2\n(median across all categories, QPS, N)")
ax.set_title("TTFT moves in opposite directions on 1 GPU, depending on scheduler")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("graphs/tp_ttft_direction_flip.png", dpi=200)
