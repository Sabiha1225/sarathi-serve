import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

s2 = pd.read_csv("custom_summary_metrics.csv").rename(columns=lambda c: c.strip())
s1 = pd.read_csv("custom_summary_metrics_tp1_pp1.csv").rename(columns=lambda c: c.strip())

models = ["llama2_7b", "mistral_7b", "qwen_7b", "yi_6b", "mixtral_7b_8expert"]

def gap(df):
    sub = df[(df["Number of requests"] == 10) & (df.QPS == 5) &
             (df["Input Output data Type"] == "long input short output")]
    p = sub.pivot(index="Model Name", columns="Scheduler", values="P95 TPOT")
    return (p["vllm"] / p["sarathi"]).reindex(models)

g2, g1 = gap(s2), gap(s1)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = range(len(models))
width = 0.35
ax.bar([i - width / 2 for i in x], g2, width, label="TP-2,PP-1 (2 GPUs)", color="#2a78d6")
ax.bar([i + width / 2 for i in x], g1, width, label="TP-1,PP-1 (1 GPU)", color="#eb6834")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xticks(list(x))
ax.set_xticklabels(models, rotation=25, ha="right")
ax.set_ylabel("vLLM P95 TPOT / sarathi P95 TPOT\n(higher = chunking helps more)")
ax.set_title("Sarathi's TPOT advantage over vLLM: TP-2 vs TP-1\n(long in/short out, N=10, QPS=5)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("graphs/tp_chunking_benefit.png", dpi=200)
