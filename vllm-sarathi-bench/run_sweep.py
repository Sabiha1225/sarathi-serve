#!/usr/bin/env python
"""
Single-config driver for the model x gpu_util x chunking x offloading sweep.

Reads ONE base YAML (configs/sweep_base.yml) describing shared settings, the list of
models, and the axis values to sweep. For every combination it builds an in-memory
config dict, writes it as a per-run provenance copy under outputs/<run_name>/config.yml
(auto-generated -- not something you maintain by hand), and shells out to
run_sarathi_style_vllm.py --config <that file> in a fresh subprocess. A fresh subprocess
per run is used deliberately: vLLM engines don't tear down cleanly enough to safely
reuse one process across different models/TP configs.

Usage:
    python run_sweep.py --base-config configs/sweep_base.yml
    python run_sweep.py --models mistral_7b llama2_7b --gpu-utils 0.45 0.25
    python run_sweep.py --chunk on --offload on --dry-run
"""
# This will run all combinations
# nohup python vllm-sarathi-bench/run_sweep.py --base-config vllm-sarathi-bench/configs/sweep_base.yml > log/sweep.log 2>&1 &

# For sarathi
# nohup python -u vllm-sarathi-bench/run_sweep.py \
#  --base-config vllm-sarathi-bench/configs/sweep_base.yml \
#  --chunk on --offload on \
#  > log/sweep_chunk_on_offload_on.log 2>&1 &

# nohup numactl -m0 -- python -u vllm-sarathi-bench/run_sweep.py \
#   --base-config vllm-sarathi-bench/configs/sweep_base.yml \
#   --chunk on --offload on \
#   > log/sweep_chunk_on_offload_on.log 2>&1 &
# echo $! > log/sweep_chunk_on_offload_on.pid

# nohup numactl -m0 -- python -u vllm-sarathi-bench/run_sweep.py \
#   --base-config vllm-sarathi-bench/configs/sweep_base.yml \
#   --chunk on --offload off \
#   > log/sweep_chunk_on_offload_off.log 2>&1 &


# For vLLM
# nohup python -u vllm-sarathi-bench/run_sweep.py \
#   --base-config vllm-sarathi-bench/configs/sweep_base.yml \
#   --chunk off --offload on \
#   > log/sweep_chunk_off_offload_on.log 2>&1 &
# echo $! > log/sweep_chunk_off_offload_on.pid

# nohup numactl -m0 -- python -u vllm-sarathi-bench/run_sweep.py \
#   --base-config vllm-sarathi-bench/configs/sweep_base.yml \
#   --chunk off --offload on \
#   > log/sweep_chunk_off_offload_on.log 2>&1 &

# nohup numactl -m0 -- python -u vllm-sarathi-bench/run_sweep.py \
#   --base-config vllm-sarathi-bench/configs/sweep_base.yml \
#   --chunk off --offload off \
#   > log/sweep_chunk_off_offload_off.log 2>&1 &

import argparse
import copy
import os
import subprocess
import sys
import time

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(REPO, "run_sarathi_style_vllm.py")


def gpu_label(gpu_util: float) -> str:
    return f"{gpu_util * 10:g}".replace(".", "_")


def build_config(base, model_spec, gpu_util, chunked, offload):
    cfg = {
        "seed": base["seed"],
        "trace_request_length_generator": base["trace_request_length_generator"],
        "request_generator": base["request_generator"],
        "poisson_request_interval_generator": base["poisson_request_interval_generator"],
    }

    model = {**base.get("model_defaults", {})}
    model["name"] = model_spec["name"]
    model["tokenizer"] = model_spec.get("tokenizer", model_spec["name"])
    model["max_model_len"] = base["max_model_len"]
    model["gpu_memory_utilization"] = gpu_util
    if model_spec.get("hf_overrides"):
        model["hf_overrides"] = copy.deepcopy(model_spec["hf_overrides"])
    cfg["model"] = model

    sched = {**base.get("vllm_scheduler_defaults", {})}
    sched["enable_chunked_prefill"] = chunked
    sched["max_num_batched_tokens"] = (
        base["chunk_size_on"] if chunked else base["max_model_len"]
    )
    cfg["vllm_scheduler"] = sched

    cfg["offloading"] = {
        "swap_space": 16,
        "cpu_offload_gb": 0,  # model-weight offload, intentionally unused here
        "kv_offloading_size": base["kv_offloading_size_on"] if offload else None,
        "kv_offloading_backend": base["kv_offloading_backend"] if offload else None,
    }

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=os.path.join(REPO, "configs", "sweep_base.yml"))
    parser.add_argument("--output-root", default=None, help="Overrides output_root from the base config.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to run (default: all in base config).")
    parser.add_argument("--gpu-utils", nargs="*", type=float, default=None, help="Subset of gpu_memory_utilization values to run.")
    parser.add_argument("--chunk", choices=["on", "off", "both"], default="both")
    parser.add_argument("--offload", choices=["on", "off", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them.")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base = yaml.safe_load(f)

    output_root = args.output_root or os.path.join(REPO, base.get("output_root", "outputs"))
    os.makedirs(output_root, exist_ok=True)

    models = base["models"]
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m["key"] in wanted]
        missing = wanted - {m["key"] for m in models}
        if missing:
            sys.exit(f"Unknown model key(s): {sorted(missing)}")

    gpu_utils = args.gpu_utils or base["gpu_memory_utilization_values"]
    chunk_values = [False, True] if args.chunk == "both" else [args.chunk == "on"]
    offload_values = [False, True] if args.offload == "both" else [args.offload == "on"]

    plan = []
    for model_spec in models:
        for gpu_util in gpu_utils:
            for chunked in chunk_values:
                for offload in offload_values:
                    chunk_label = "chunk_on" if chunked else "chunk_off"
                    offload_label = "offload_on" if offload else "offload_off"
                    run_name = f"{model_spec['key']}_gpu_{gpu_label(gpu_util)}_{chunk_label}_{offload_label}"
                    plan.append((run_name, model_spec, gpu_util, chunked, offload))

    print(f"Planned {len(plan)} runs.")
    if args.dry_run:
        for run_name, *_ in plan:
            print(" ", run_name)
        return

    for run_name, model_spec, gpu_util, chunked, offload in plan:
        run_dir = os.path.join(output_root, run_name)

        if os.path.exists(os.path.join(run_dir, "results.csv")):
            print(f"[SKIP] Already exists: {run_dir}")
            continue

        os.makedirs(run_dir, exist_ok=True)

        cfg = build_config(base, model_spec, gpu_util, chunked, offload)
        cfg["output_csv"] = os.path.join(run_dir, "results.csv")
        cfg["dump_requests_jsonl"] = os.path.join(run_dir, "requests.jsonl")
        cfg["prometheus_metrics_jsonl"] = os.path.join(run_dir, "prometheus_metrics.jsonl")
        
        cfg_path = os.path.join(run_dir, "config.yml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print()
        print("=" * 60)
        print(f"Run: {run_name}")
        print("=" * 60)

        gpu_log = open(os.path.join(run_dir, "nvidia-smi-output.csv"), "w")
        top_log = open(os.path.join(run_dir, "top_output.txt"), "w")
        run_log_path = os.path.join(run_dir, "run.log")

        gpu_proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
             "--format=csv", "-lms", "250"],
            stdout=gpu_log, stderr=subprocess.STDOUT,
        )
        top_proc = subprocess.Popen(
            ["top", "-d", "4.0", "-u", os.environ.get("USER", ""), "-b"],
            stdout=top_log, stderr=subprocess.STDOUT,
        )

        try:
            with open(run_log_path, "w") as run_log:
                proc = subprocess.run(
                    [sys.executable, RUNNER, "--config", cfg_path],
                    stdout=run_log, stderr=subprocess.STDOUT,
                )
                status = proc.returncode
        finally:
            for p in (gpu_proc, top_proc):
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
            gpu_log.close()
            top_log.close()

        with open(os.path.join(run_dir, "run_summary.txt"), "w") as f:
            f.write(f"model={model_spec['key']}\n")
            f.write(f"gpu_memory_utilization={gpu_util}\n")
            f.write(f"chunked_prefill={chunked}\n")
            f.write(f"kv_offloading={offload}\n")
            f.write(f"status={status}\n")

        if status == 0:
            print(f"[SUCCESS] {run_name}")
        else:
            failed_dir = f"{run_dir}_failed_{time.strftime('%Y%m%d_%H%M%S')}"
            os.rename(run_dir, failed_dir)
            print(f"[ERROR] Run failed: {run_name} (status={status})")

    print()
    print("Sweep completed.")


if __name__ == "__main__":
    main()
