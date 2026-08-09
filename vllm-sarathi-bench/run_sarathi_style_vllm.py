import argparse, asyncio, csv, inspect, json, math, random, time
import numpy as np
import pandas as pd
import yaml
import dataclasses
from vllm.v1.metrics.reader import get_metrics_snapshot


def make_requests(cfg):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    tcfg = cfg["trace_request_length_generator"]
    df = pd.read_csv(tcfg["trace_file"])

    df["num_prefill_tokens"] = (
        df["num_prefill_tokens"] * tcfg["prefill_scale_factor"]
    ).astype(int)
    df["num_decode_tokens"] = (
        df["num_decode_tokens"] * tcfg["decode_scale_factor"]
    ).astype(int)

    total = df["num_prefill_tokens"] + df["num_decode_tokens"]
    diff = (total - tcfg["max_tokens"]).clip(lower=0)

    prefill_ratio = df["num_prefill_tokens"] / total
    decode_ratio = df["num_decode_tokens"] / total

    df["num_prefill_tokens"] -= np.ceil(diff * prefill_ratio).astype(int)
    df["num_decode_tokens"] -= np.ceil(diff * decode_ratio).astype(int)

    df["num_prefill_tokens"] = df["num_prefill_tokens"].clip(lower=1)
    df["num_decode_tokens"] = df["num_decode_tokens"].clip(lower=1)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.iloc[: cfg["request_generator"]["num_requests"]]

    qps = cfg["poisson_request_interval_generator"]["qps"]
    max_interval = 3.0 / qps
    now = 0.0
    requests = []

    # Safe vocab range across all 5 models in the sweep (Llama-2 and the
    # Mixtral variant have the smallest vocab at 32000; staying well under
    # that avoids embedding-index-out-of-range errors on any of them).
    # 0/1 skipped since those are commonly special tokens (pad/bos) in most
    # tokenizers, though harmless either way with dummy weights.
    vocab_low = cfg["request_generator"].get("random_prompt_vocab_low", 2)
    vocab_high = cfg["request_generator"].get("random_prompt_vocab_high", 30000)

    for i, row in df.iterrows():
        interval = -math.log(1.0 - random.random()) / qps
        now += min(interval, max_interval)
        num_prefill = int(row["num_prefill_tokens"])
        # Unique random content per request -> an offload-cache hit can only
        # come from this exact request recovering its own evicted blocks,
        # never from colliding with another request's identical content.
        prompt_token_ids = np.random.randint(vocab_low, vocab_high, size=num_prefill).tolist()
        requests.append({
            "id": i,
            "arrived_at": now,
            "num_prefill_tokens": num_prefill,
            "num_decode_tokens": int(row["num_decode_tokens"]),
            "prompt_token_ids": prompt_token_ids,
        })

    return requests


def make_engine_args(cfg):
    from vllm.engine.arg_utils import AsyncEngineArgs

    model = cfg["model"]
    sched = cfg["vllm_scheduler"]
    offload = cfg["offloading"]

    kwargs = {
        "model": model["name"],
        "tokenizer": model.get("tokenizer", model["name"]),
        "dtype": model["dtype"],
        "tensor_parallel_size": model["tensor_parallel_size"],
        "pipeline_parallel_size": model["pipeline_parallel_size"],
        "max_model_len": model["max_model_len"],
        "gpu_memory_utilization": model["gpu_memory_utilization"],
        "trust_remote_code": model.get("trust_remote_code", True),
        "max_num_seqs": sched["max_num_seqs"],
        "max_num_batched_tokens": sched["max_num_batched_tokens"],
        "disable_hybrid_kv_cache_manager": sched.get("disable_hybrid_kv_cache_manager"),
        "enable_chunked_prefill": sched.get("enable_chunked_prefill"),
        "enable_prefix_caching": sched.get("enable_prefix_caching"),
        "swap_space": offload.get("swap_space"),
        "cpu_offload_gb": offload.get("cpu_offload_gb"),
        "kv_offloading_backend": offload.get("kv_offloading_backend"),
        "disable_log_requests": True,
        "attention_backend": model.get("attention_backend"),
        "hf_overrides": model.get("hf_overrides"),
        "load_format": model.get("load_format"),
        "enforce_eager": model.get("enforce_eager"),
        "kv_offloading_size": offload.get("kv_offloading_size"),
        "kv_offloading_backend": offload.get("kv_offloading_backend"),
        "kv_cache_metrics": True,
        "kv_cache_metrics_sample": 1.0,
    }

    valid = inspect.signature(AsyncEngineArgs).parameters
    return AsyncEngineArgs(**{k: v for k, v in kwargs.items() if k in valid and v is not None})


async def run_one(engine, req, start_time):
    from vllm import SamplingParams

    delay = start_time + req["arrived_at"] - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)

    submit = time.monotonic()
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=req["num_decode_tokens"],
        ignore_eos=True,
    )

    stream = engine.generate(
        {"prompt_token_ids": req["prompt_token_ids"]},
        params,
        str(req["id"]),
    )

    first = None
    final = None
    async for out in stream:
        final = out
        if first is None and out.outputs and len(out.outputs[0].token_ids) > 0:
            first = time.monotonic()

    end = time.monotonic()
    generated = len(final.outputs[0].token_ids) if final and final.outputs else 0

    return {
        "request_id": req["id"],
        "arrival_s": req["arrived_at"],
        "prompt_tokens": req["num_prefill_tokens"],
        "requested_decode_tokens": req["num_decode_tokens"],
        "generated_tokens": generated,
        "ttft_s": None if first is None else first - submit,
        "latency_s": end - submit,
        "finish_s": end - start_time,
    }


async def snapshot_prometheus_metrics(output_path, start_time, interval_s=5.0):
    with open(output_path, "w") as f:
        while True:
            snap = {
                "t_s": time.monotonic() - start_time,
                "metrics": [
                    {**dataclasses.asdict(m), "kind": type(m).__name__}
                    for m in get_metrics_snapshot()
                ],
            }
            f.write(json.dumps(snap) + "\n")
            f.flush()
            await asyncio.sleep(interval_s)

async def main_async(cfg):
    try:
        from vllm import AsyncLLMEngine
    except ImportError:
        from vllm.engine.async_llm_engine import AsyncLLMEngine

    requests = make_requests(cfg)

    if cfg.get("dump_requests_jsonl"):
        with open(cfg["dump_requests_jsonl"], "w") as f:
            for r in requests:
                r_light = {k: v for k, v in r.items() if k != "prompt_token_ids"}
                f.write(json.dumps(r_light) + "\n")

    engine = AsyncLLMEngine.from_engine_args(make_engine_args(cfg))
    print(f"[TIMING] request processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.monotonic()

    metrics_task = None
    if cfg.get("prometheus_metrics_jsonl"):
        metrics_task = asyncio.create_task(
            snapshot_prometheus_metrics(cfg["prometheus_metrics_jsonl"], start)
        )

    tasks = [asyncio.create_task(run_one(engine, r, start)) for r in requests]
    results = [await t for t in asyncio.as_completed(tasks)]
    results.sort(key=lambda x: x["arrival_s"])

    end_to_end_s = time.monotonic() - start
    print(
        f"[TIMING] request processing finished at {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"-- end_to_end_s={end_to_end_s:.3f}"
    )

    if metrics_task:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

    with open(cfg["output_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
