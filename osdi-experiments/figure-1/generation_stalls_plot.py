import pandas as pd
import matplotlib.pyplot as plt
import yaml
import uuid
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pathlib import Path
from typing import Dict

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

#OUTPUT_DIR = f"{Path.cwd()}/benchmark_output"

def _get_run_directories():
    rootdir = Path(OUTPUT_DIR)
    subdirectory_list = [
        directory for directory in rootdir.iterdir() if directory.is_dir()
    ]
    return subdirectory_list

def _get_decode_completion_times():
    #run_directories = _get_run_directories()
    run_directories = []
    # run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/gen_stall_sarathi")
    # run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/gen_stall_sarathi_arxiv")
    # run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/gen_stall_sarathi_arxiv_1")
    # run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/gen_stall_sarathi_adaptive_chunk")
    # run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/gen_stall_sarathi_adaptive_chunk_arxiv")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/2026-04-20_22-34-14-379606")
    # run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/gen_stall_vllm_1")
    run_directories.append("/home/sabiha/sarathi-serve/benchmark_output/figure-1/gen_stall_vllm_arxiv")
    run_directories.sort()

    datapoints = []

    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    for run_dir in run_directories:
        try:
            with open(f"{run_dir}/replica_0/config.yml", "r") as config_file, open(
                f"{run_dir}/benchmark_config.yml", "r"
            ) as benchmark_config_file, open(
                f"{run_dir}/replica_0/plots/decode_completion_time_series.csv", "r"
            ) as decode_completion_times_file:
                config = yaml.safe_load(config_file)
                benchmark_config = yaml.safe_load(benchmark_config_file)
                decode_completion_times = pd.read_csv(decode_completion_times_file)

                scheduler = config["scheduler_type"]
                if config["scheduler_type"] == "sarathi":
                    scheduler += f"_{config['chunk_size']}"
                
                decode_completion_times["name"] = scheduler
                datapoints.append(decode_completion_times)
        except FileNotFoundError as e:
            print(f"WARN: Skipping {run_dir} due to {e}")
    
    return pd.concat(datapoints)

def plot():
    df = _get_decode_completion_times()
    # Group the data by the 'id' column
    groups = df.groupby("name")

    # Create a new figure
    fig, ax = plt.subplots()

    # Iterate over each group
    for name, group in groups:
        # Plot the time series for this group
        color = "chocolate" if name == "vllm" else "green"
        ax.plot(group["Time (sec)"],
                group["decode_completion"],
                label=name,
                linewidth=1.5,
                color=color)

    # Add a legend
    ax.legend(loc="upper left", fontsize=22)

    # Label the x and y axes
    ax.set_xlabel("Time (s)", fontweight="bold", fontsize=20)
    ax.set_ylabel("# output tokens", fontweight="bold", fontsize=22)

    # Add gridlines
    ax.grid(True, linestyle="--")

    plt.yticks(fontsize=12)
    plt.xticks(fontsize=11)

    ylabels = ["{:,.0f}".format(y) + "K" for y in ax.get_yticks() / 1000]
    ax.set_yticklabels(ylabels)

    # Create a set of inset Axes: these should fill the bounding box allocated to them.
    ax_sub = ax.inset_axes([0.625, 0.125, 0.33, 0.33])
    for name, group in groups:
        color = "chocolate" if name == "vllm" else "green"
        ax_sub.plot(group["Time (sec)"],
                    group["decode_completion"],
                    label=name,
                    marker=".",
                    color=color)
    ax_sub.set_xlim(225, 275)  # specify the limits for x-axis
    ax_sub.set_ylim(10000, 12500)  # specify the limits for y-axis

    # Remove the y-axis ticks from the zoomed subplot
    ax_sub.yaxis.set_ticks([])

    # Add lines connecting the zoomed subplot to the main plot
    mark_inset(ax, ax_sub, loc1=3, loc2=1, fc="none", ec="0.5")

    # Show the plot
    # plt.savefig("yi-arxiv.pdf")
    # plt.savefig("yi-arxiv.png")
    # plt.savefig("llama-sharegpt-adaptive-chunking.pdf")
    # plt.savefig("llama-sharegpt-adaptive-chunking.png")
    # plt.savefig("llama-sharegpt-adaptive-chunking_arxiv.pdf")
    # plt.savefig("llama-sharegpt-adaptive-chunking_arxiv.png")
    plt.savefig("llama-adaptive-chunking_arxiv.pdf")
    plt.savefig("llama-adaptive-chunking_arxiv.png")
    # plt.savefig("llama-arxiv-sarathi.pdf")
    # plt.savefig("llama-arxiv-sarathi.png")
    plt.show()

plot()