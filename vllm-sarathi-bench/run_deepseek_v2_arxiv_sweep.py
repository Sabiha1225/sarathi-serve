#!/usr/bin/env python

import argparse
import copy
import csv
import os
import subprocess
import sys
import time

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_sarathi_style_vllm.py")

# nohup python -u vllm-sarathi-bench/run_deepseek_v2_arxiv_sweep.py \
#   --base-config vllm-sarathi-bench/configs/deepseek_v2_arxiv.yml \
#   > log/deepseek_v2_arxiv_tp2_pp1.log 2>&1 &


# nohup env \
#   LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}" \
#   LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
#   python -u vllm-sarathi-bench/run_deepseek_v2_arxiv_sweep.py \
#   --base-config vllm-sarathi-bench/configs/deepseek_v2_arxiv.yml \
#   > log/deepseek_v2_arxiv_tp2_pp1.log 2>&1 &

# tail -f /home/sabiha/sarathi-serve/log/deepseek_v2_arxiv_sweep.log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default=os.path.join(
            HERE,
            "configs",
            "deepseek_v2_arxiv.yml",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base = yaml.safe_load(f)

    output_root = base["output_root"]
    os.makedirs(output_root, exist_ok=True)

    summary_path = os.path.join(output_root, "sweep_summary.csv")

    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model",
                "qps",
                "chunk_size",
                "status",
                "wall_time_s",
                "output_dir",
            ])

    plan = [
        (qps, chunk_size)
        for qps in base["qps_values"]
        for chunk_size in base["chunk_sizes"]
    ]

    print(f"Planned {len(plan)} runs")

    for qps, chunk_size in plan:
        run_name = (
            f"deepseek_v2_lite"
            f"_qps_{qps}"
            f"_chunk_{chunk_size}"
            f"_offload_off"
        )
        run_dir = os.path.join(output_root, run_name)
        results_path = os.path.join(run_dir, "results.csv")

        if os.path.exists(results_path):
            print(f"[SKIP] {run_name}")
            continue

        if args.dry_run:
            print(
                f"[DRY RUN] qps={qps}, "
                f"chunk_size={chunk_size}, "
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
            "model": copy.deepcopy(base["model"]),
            "vllm_scheduler": copy.deepcopy(
                base["vllm_scheduler"]
            ),
            "offloading": copy.deepcopy(base["offloading"]),
            "output_csv": results_path,
            "dump_requests_jsonl": os.path.join(
                run_dir, "requests.jsonl"
            ),
            "prometheus_metrics_jsonl": os.path.join(
                run_dir, "prometheus_metrics.jsonl"
            ),
        }

        # In vLLM, this is the total scheduled-token budget per iteration.
        # With chunked prefill enabled, long prefills are divided according
        # to the available budget.
        cfg["vllm_scheduler"]["max_num_batched_tokens"] = int(
            chunk_size
        )

        config_path = os.path.join(run_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        run_log_path = os.path.join(run_dir, "run.log")
        gpu_log_path = os.path.join(
            run_dir, "nvidia-smi-output.csv"
        )

        print()
        print("=" * 70)
        print(f"Run: {run_name}")
        print(f"QPS: {qps}")
        print(f"Chunk budget: {chunk_size}")
        print("KV-cache offloading: disabled")
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

            start = time.monotonic()

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
                status = process.returncode
            finally:
                gpu_monitor.terminate()
                try:
                    gpu_monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    gpu_monitor.kill()
                    gpu_monitor.wait()

        wall_time = time.monotonic() - start
        status_text = "success" if status == 0 else "failed"

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                cfg["model"]["name"],
                qps,
                chunk_size,
                status_text,
                f"{wall_time:.3f}",
                run_dir,
            ])

        if status == 0:
            print(
                f"[SUCCESS] {run_name}: "
                f"{wall_time:.3f} seconds"
            )
        else:
            print(
                f"[FAILED] {run_name}: status={status}. "
                f"See {run_log_path}"
            )
            sys.exit(status)

    print()
    print(f"Sweep completed. Summary: {summary_path}")


if __name__ == "__main__":
    main()