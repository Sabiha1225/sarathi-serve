import pandas as pd
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
from pathlib import Path
from typing import Dict

OUTPUT_DIR = f"{Path.cwd()}/high_tail_latency_output"

def _get_run_directories():
    rootdir = Path(OUTPUT_DIR)
    subdirectory_list = [
        directory for directory in rootdir.iterdir() if directory.is_dir()
    ]
    return subdirectory_list

def _get_df():
    #run_directories = _get_run_directories()
    run_directories = []
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/high_tail_latency_sarathi_1")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/high_tail_latency_sarathi_7")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/high_tail_latency_sarathi_55")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/high_tail_latency_vllm_1")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/high_tail_latency_vllm_7")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/high_tail_latency_vllm_55")
    
    run_directories.sort()

    datapoints = []

    for run_dir in run_directories:
        try:
            with open(
                f"{run_dir}/benchmark_config.yml", "r"
            ) as benchmark_config_file, open(
                f"{run_dir}/replica_0/plots/decode_token_execution_plus_preemption_time.csv", "r"
            ) as tbt_file:
                benchmark_config = yaml.safe_load(benchmark_config_file)
                tbt_df = pd.read_csv(tbt_file)

                scheduler = benchmark_config["replica_scheduler_provider"]
                if scheduler == "sarathi":
                    scheduler += f"_{benchmark_config['sarathi_scheduler_chunk_size']}"
                qps = benchmark_config["poisson_request_interval_generator_qps"]
                
                datapoints.append({
                    "scheduler": scheduler,
                    "qps": qps,
                    "tail_latency": tbt_df["decode_token_execution_plus_preemption_time"].quantile(0.99)
                })
        except FileNotFoundError as e:
            print(f"WARN: Skipping {run_dir} due to {e}")
    return pd.DataFrame(datapoints)

def plot():
    df = _get_df()
    # sns.set_theme(style="whitegrid")
    sns.barplot(data=df, x='qps', y='tail_latency', hue='scheduler')
    plt.savefig("high_tail_latency.pdf")
    plt.savefig("high_tail_latency.png")
    plt.show()

plot()