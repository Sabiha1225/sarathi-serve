#!/usr/bin/env python3

"""
Run the DeepSeek-V2-Lite ARXIV benchmark using standard vLLM
with chunked prefill disabled.

Sweep dimension:
    QPS = 1, 5, 10, 15, 20

Fixed settings:
    enable_chunked_prefill = False
    max_num_batched_tokens = max_model_len
    KV-cache offloading = disabled
    model-weight CPU offloading = disabled
"""


# nohup env \
#   LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}" \
#   LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
#   python -u \
#   vllm-sarathi-bench/run_deepseek_v2_arxiv_vllm_sweep.py \
#   --base-config \
#   vllm-sarathi-bench/configs/deepseek_v2_arxiv_vllm.yml \
#   > log/deepseek_v2_arxiv_chunk_off_tp2_pp1.log 2>&1 &

import argparse
import copy
import csv
import os
import subprocess
import sys
import time

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))

RUNNER = os.path.join(
    HERE,
    "run_sarathi_style_vllm.py",
)

DEFAULT_CONFIG = os.path.join(
    HERE,
    "configs",
    "deepseek_v2_arxiv_vllm.yml",
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-config",
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.base_config) as f:
        base = yaml.safe_load(f)

    output_root = base["output_root"]
    os.makedirs(output_root, exist_ok=True)

    summary_path = os.path.join(
        output_root,
        "sweep_summary.csv",
    )

    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model",
                "qps",
                "chunked_prefill",
                "max_num_batched_tokens",
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "status",
                "wall_time_s",
                "output_dir",
            ])

    qps_values = base["qps_values"]

    print(f"Planned {len(qps_values)} runs")
    print("Chunked prefill: disabled")
    print(
        "max_num_batched_tokens: "
        f"{base['vllm_scheduler']['max_num_batched_tokens']}"
    )

    for qps in qps_values:
        run_name = (
            f"deepseek_v2_lite"
            f"_qps_{qps}"
            f"_chunk_off"
            f"_offload_off"
        )

        run_dir = os.path.join(
            output_root,
            run_name,
        )

        results_path = os.path.join(
            run_dir,
            "results.csv",
        )

        if os.path.exists(results_path):
            print(f"[SKIP] Already completed: {run_name}")
            continue

        if args.dry_run:
            print(
                f"[DRY RUN] qps={qps}, "
                f"chunked_prefill=False, "
                f"output={run_dir}"
            )
            continue

        os.makedirs(run_dir, exist_ok=True)

        cfg = {
            "seed": base["seed"],

            "trace_request_length_generator": copy.deepcopy(
                base["trace_request_length_generator"]
            ),

            "request_generator": copy.deepcopy(
                base["request_generator"]
            ),

            "poisson_request_interval_generator": {
                "qps": float(qps),
            },

            "model": copy.deepcopy(
                base["model"]
            ),

            "vllm_scheduler": copy.deepcopy(
                base["vllm_scheduler"]
            ),

            "offloading": copy.deepcopy(
                base["offloading"]
            ),

            "output_csv": results_path,

            "dump_requests_jsonl": os.path.join(
                run_dir,
                "requests.jsonl",
            ),

            "prometheus_metrics_jsonl": os.path.join(
                run_dir,
                "prometheus_metrics.jsonl",
            ),
        }

        # Enforce the intended non-chunked configuration.
        cfg["vllm_scheduler"][
            "enable_chunked_prefill"
        ] = False

        cfg["vllm_scheduler"][
            "max_num_batched_tokens"
        ] = int(cfg["model"]["max_model_len"])

        # Explicitly enforce no CPU/KV-cache offloading.
        cfg["offloading"]["cpu_offload_gb"] = 0
        cfg["offloading"]["kv_offloading_size"] = None
        cfg["offloading"]["kv_offloading_backend"] = None

        config_path = os.path.join(
            run_dir,
            "config.yml",
        )

        with open(config_path, "w") as f:
            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,
            )

        run_log_path = os.path.join(
            run_dir,
            "run.log",
        )

        gpu_log_path = os.path.join(
            run_dir,
            "nvidia-smi-output.csv",
        )

        print()
        print("=" * 70)
        print(f"Run: {run_name}")
        print(f"QPS: {qps}")
        print("Chunked prefill: disabled")
        print(
            "max_num_batched_tokens: "
            f"{cfg['vllm_scheduler']['max_num_batched_tokens']}"
        )
        print("KV-cache offloading: disabled")
        print("Model-weight CPU offloading: disabled")
        print("=" * 70)

        with open(gpu_log_path, "w") as gpu_log:
            gpu_monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu="
                    "timestamp,index,memory.used,"
                    "utilization.gpu",
                    "--format=csv",
                    "-lms",
                    "250",
                ],
                stdout=gpu_log,
                stderr=subprocess.STDOUT,
            )

            start_time = time.monotonic()

            try:
                with open(run_log_path, "w") as run_log:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            RUNNER,
                            "--config",
                            config_path,
                        ],
                        stdout=run_log,
                        stderr=subprocess.STDOUT,
                    )

                return_code = process.returncode

            finally:
                gpu_monitor.terminate()

                try:
                    gpu_monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    gpu_monitor.kill()
                    gpu_monitor.wait()

        wall_time_s = time.monotonic() - start_time

        status = (
            "success"
            if return_code == 0
            else "failed"
        )

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                cfg["model"]["name"],
                qps,
                False,
                cfg["vllm_scheduler"][
                    "max_num_batched_tokens"
                ],
                cfg["model"]["tensor_parallel_size"],
                cfg["model"]["pipeline_parallel_size"],
                status,
                f"{wall_time_s:.3f}",
                run_dir,
            ])

        if return_code == 0:
            print(
                f"[SUCCESS] {run_name}: "
                f"{wall_time_s:.3f} seconds"
            )
        else:
            print(
                f"[FAILED] {run_name}: "
                f"return_code={return_code}"
            )
            print(f"See log: {run_log_path}")
            sys.exit(return_code)

    print()
    print("Sweep completed.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()