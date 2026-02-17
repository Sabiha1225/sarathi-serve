import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


from pathlib import Path
from typing import Dict

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

SEQUENCE_LENGTHS = [2048, 4096, 8192, 16384]
# CHUNK_SIZES = [512, 1024, 2048] + [max(SEQUENCE_LENGTHS)]
CHUNK_SIZES = [512, 1024, 2048]

def _process_run(
    benchmark_config: Dict[str, object],
    batch_metrics: pd.DataFrame,
):
    prefill_length = benchmark_config["uniform_request_length_generator_max_tokens"] - 1
    chunk_size = benchmark_config["sarathi_scheduler_chunk_size"]
    num_requests = benchmark_config["synthetic_request_generator_num_requests"]

    batch_metrics = batch_metrics[batch_metrics["batch_num_decode_tokens"] == 0]
    prefill_execution_time = batch_metrics["batch_execution_time"].sum() * 1000
    print(f"prefill_length {prefill_length} chunk_size {chunk_size}")
    return {
        "prefill_length": prefill_length,
        "chunk_size": str(chunk_size),
        "prefill_execution_time": prefill_execution_time / num_requests,
    }

# Process runs to find throughput gain and decode speedup
def _process_runs():
    # run_directories = _get_run_directories()
    run_directories = []
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_512_uni_token_2048")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_512_uni_token_4096")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_512_uni_token_8192")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_512_uni_token_16384")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_1024_uni_token_2048")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_1024_uni_token_4096")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_1024_uni_token_8192")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_1024_uni_token_16384")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_2048_uni_token_2048")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_2048_uni_token_4096")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_2048_uni_token_8192")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_2048_uni_token_16384")
    #run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_16384_uni_token_2048")
    #run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_16384_uni_token_4096")
    #run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_16384_uni_token_8192")
    #run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-9a/chunk_16384_uni_token_16384")
    
    run_directories.sort()
    num_runs = len(SEQUENCE_LENGTHS) * len(CHUNK_SIZES)
    run_directories = run_directories[-num_runs:]

    baseline = {}
    datapoints = []

    yaml.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)
    for run_dir in run_directories:
        try:
            with open(
                f"{run_dir}/benchmark_config.yml", "r"
            ) as benchmark_config_file, open(
                f"{run_dir}/replica_0/batch_metrics.csv", "r"
            ) as batch_metrics_file:
                yaml.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)
                benchmark_config = yaml.safe_load(benchmark_config_file)
                batch_metrics = pd.read_csv(batch_metrics_file)
                datapoint = _process_run(benchmark_config, batch_metrics)
                # if int(datapoint["chunk_size"]) == max(SEQUENCE_LENGTHS):
                if int(datapoint["chunk_size"]) == SEQUENCE_LENGTHS[0]:
                    baseline[datapoint["prefill_length"]] = datapoint.copy()
                datapoints.append(datapoint)
        except FileNotFoundError as e:
            print(f"Skipping {run_dir} due to {e}")

    print(baseline)
    for datapoint in datapoints:
        datapoint["prefill_execution_time_relative"] = (
            datapoint["prefill_execution_time"]
            / baseline[datapoint["prefill_length"]]["prefill_execution_time"]
        )
    return pd.DataFrame(datapoints)

def plot():
    df = _process_runs()
    df.to_csv("chunking-overhead.csv", index=False)

    sns.set_style("whitegrid")
    g = sns.FacetGrid(df, col='prefill_length', sharex=False, sharey=True)
    g.map(sns.barplot, "chunk_size", 'prefill_execution_time_relative')
    plt.savefig("prefill_chunking_overhead_plot.png")
    plt.show()

plot()