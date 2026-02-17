import pandas as pd
import yaml

from pathlib import Path

TP_DIMENSION = 2
MAX_BATCH_SIZE = 4
NUM_REQUESTS = 2 * MAX_BATCH_SIZE
PREFILL_LENGTH = 1024
DECODE_LENGTH = 16
SEQUENCE_LENGTH = PREFILL_LENGTH + DECODE_LENGTH

def _filter_df (df: pd.DataFrame):
    df = df[
        ((df["batch_num_prefill_tokens"] == PREFILL_LENGTH) & (df["batch_num_decode_tokens"] == 0) & (df["batch_size"] == 1))
        | ((df["batch_num_prefill_tokens"] == 0) & (df["batch_num_decode_tokens"] == MAX_BATCH_SIZE) & (df["batch_size"] == MAX_BATCH_SIZE))
        | ((df["batch_num_prefill_tokens"] == PREFILL_LENGTH - MAX_BATCH_SIZE + 1) & (df["batch_num_decode_tokens"] == MAX_BATCH_SIZE - 1) & (df["batch_size"] == MAX_BATCH_SIZE))
    ]
    df = df.groupby(["batch_num_prefill_tokens", "batch_num_decode_tokens", "batch_size"]).median().reset_index()
    return df

def get_df():
    # run_directories = _get_run_directories()
    run_directories = []
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/table-2/sarathi_with_op")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/table-2/sarathi_without_op")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/table-2/vllm_with_op")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/table-2/vllm_without_op")
    run_directories.sort()

    non_profiled_dfs = []
    profiled_dfs = [] 
    for run_dir in run_directories:
        try:
            with open(
                f"{run_dir}/benchmark_config.yml", "r"
            ) as benchmark_config_file, open(
                f"{run_dir}/replica_0/batch_metrics.csv", "r"
            ) as batch_metrics_file:
                benchmark_config = yaml.safe_load(benchmark_config_file)
                batch_metrics = pd.read_csv(batch_metrics_file)

                if benchmark_config["metrics_store_enable_op_level_metrics"]:
                    operation_metrics = pd.read_csv(f"{run_dir}/replica_0/operation_metrics.csv")
                    operation_metrics = pd.merge(batch_metrics, operation_metrics, on=["Batch Id"], how="inner")
                    profiled_dfs.append(operation_metrics)

                non_profiled_dfs.append(batch_metrics)
        except FileNotFoundError as e:
            print(f"WARN: Skipping {run_dir} due to {e}")
    
    non_profiled_df = _filter_df(pd.concat(non_profiled_dfs))
    non_profiled_df["batch_execution_time"] *= 1000
    non_profiled_df["per_token_time"] = non_profiled_df["batch_execution_time"] / non_profiled_df["batch_num_tokens"]
    profiled_df = _filter_df(pd.concat(profiled_dfs))
    profiled_df["linear"] = sum([profiled_df[x] for x in ["mlp_up_proj", "mlp_down_proj", "attn_pre_proj", "attn_post_proj"]])
    profiled_df["attention"] = sum([profiled_df[x] for x in ["attn"]])
    profiled_df = profiled_df[["batch_num_prefill_tokens", "batch_num_decode_tokens", "batch_size", "linear", "attention"]]

    return pd.merge(non_profiled_df, profiled_df, on=["batch_num_prefill_tokens", "batch_num_decode_tokens", "batch_size"], how="inner")

df = get_df()
df.to_csv("results.csv", index=False)
df