#!/usr/bin/env python3
"""
Grouped bar chart: total inference time at QPS 15 for five models, across
four configurations parsed directly out of Inference_Results(Sarathi_Baseline).csv:
  - Sarathi, static chunk_size=512, dynamic_chunking=False
  - Sarathi, dynamic_chunking=True
  - vLLM baseline
  - Best chunk size found by sweeping 128-4096 (bottom section of the CSV)

python3 plot_qps15_compare_baselines.py

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/sabiha/sarathi-serve/graphs_baselines_sarathi/Inference_Results(Sarathi_Baseline).csv"
QPS_TARGET = 15
MODELS = ["llama", "mistral", "mixtral", "qwen", "yi"]

raw = pd.read_csv(CSV_PATH, header=None, skiprows=1)

# the bottom chunk-size-sweep section starts at the row whose first cell is "Model"
split_idx = raw.index[raw[0] == "Model"][0]
top = raw.iloc[:split_idx].copy()
bottom = raw.iloc[split_idx + 1:].copy()

results = {}
for model in MODELS:
    # --- static 512 (cols: 1=model,2=qps,3=chunk_size,4=dynamic_chunking,5=total_time_sec) ---
    row = top[(top[1] == model) & (top[2].astype(float) == QPS_TARGET)]
    s512 = float(row[5].iloc[0])

    # --- dynamic (cols: 9=model,10=qps,12=dynamic_chunking,13=total_time_sec) ---
    row = top[(top[9] == model) & (top[10].astype(float) == QPS_TARGET)]
    dyn = float(row[13].iloc[0])

    # --- vLLM (cols: 17=model,18=qps,19=total_time_sec) ---
    row = top[(top[17] == model) & (top[18].astype(float) == QPS_TARGET)]
    vllm = float(row[19].iloc[0])

    # --- best chunk size sweep (cols: 0=Model,1=Chunk Size,2=Inference " s",4=QPS) ---
    sub = bottom[(bottom[0] == model) & (bottom[4].astype(float) == QPS_TARGET)].copy()
    sub[2] = sub[2].str.replace(" s", "", regex=False).astype(float)
    best_row = sub.loc[sub[2].idxmin()]
    best_val = float(best_row[2])
    best_chunk = int(best_row[1])

    results[model] = {"s512": s512, "dyn": dyn, "vllm": vllm, "best": best_val, "best_chunk": best_chunk}
    print(f"{model:8s} static512={s512:7.2f}  dynamic={dyn:7.2f}  vllm={vllm:7.2f}  best={best_val:7.2f} @{best_chunk}")

# ---------------------------------------------------------------- plot
SERIES = [
    ("s512", "Sarathi \u00b7 static 512", "#2a78d6"),
    ("dyn",  "Sarathi \u00b7 dynamic",     "#eb6834"),
    ("vllm", "vLLM",                       "#1baf7a"),
    ("best", "Best sweep",                 "#eda100"),
]

x = np.arange(len(MODELS))
n_series = len(SERIES)
bar_w = 0.8 / n_series

fig, ax = plt.subplots(figsize=(11, 6))
for i, (key, label, color) in enumerate(SERIES):
    vals = [results[m][key] for m in MODELS]
    offset = (i - (n_series - 1) / 2) * bar_w
    bars = ax.bar(x + offset, vals, width=bar_w, label=label, color=color)
    if key == "best":
        for b, m in zip(bars, MODELS):
            ax.annotate(f"{results[m]['best']:.1f}\n@{results[m]['best_chunk']}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.set_ylabel("total inference time (s)")
ax.set_title(f"Inference time by configuration, QPS={QPS_TARGET} (300 requests, Arxiv, TP-2/PP-1)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("qps15_compare.png", dpi=150)
print("wrote qps15_compare.png")
