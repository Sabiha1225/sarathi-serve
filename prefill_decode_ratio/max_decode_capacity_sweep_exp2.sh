#!/usr/bin/env bash
set -uo pipefail

# chmod +x prefill_decode_ratio/max_decode_capacity_sweep_exp2.sh
# nohup ./prefill_decode_ratio/max_decode_capacity_sweep_exp2.sh \
#   > log/max_decode_capacity_sweep_exp2.log 2>&1 &

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
CONSTANTS_FILE="$REPO/sarathi/benchmark/constants.py"
OUTPUT_ROOT="$REPO/benchmark_output/pd_ratio_max_decode_capacity_exp2"
TIME_FILE="$REPO/log/pd_ratio_max_decode_capacity_exp2.txt"
LOG_DIR="$REPO/log"
SUMMARY="$OUTPUT_ROOT/max_decode_capacity_summary.csv"
CONFIG_YML="default.yml"

# Pin to one GPU. This machine has two L40S; change to 1 to use the other.
GPU_ID=0
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Decode tokens per sequence — held fixed while N (concurrency) sweeps.
# 2048 stays inside every model's native max_position_embeddings (Llama-2 and
# Yi are natively 4096), so there's no RoPE-extrapolation caveat this time.
# This is the second axis the earlier answer mentioned: rerun with a
# different value here for a second series if you want to see how the max
# concurrency shrinks as sequences run longer.
DECODE_TOKENS_PER_SEQ=2048

# Concurrency sweep — this is what's actually being tested: how many
# sequences can decode at once (each contributes 1 token/iteration, so N
# concurrent sequences = N decode tokens processed per iteration).
N_VALUES=(16 32 64 128 256 512 1024 2048 4096)

# model_name|safe_label
MODELS=(
    "meta-llama/Llama-2-7b-hf|llama"
    "mistralai/Mistral-7B-Instruct-v0.3|mistral"
    "DiscoResearch/mixtral-7b-8expert|mixtral"
    "Qwen/Qwen-7B|qwen"
    "01-ai/Yi-6B|yi"
)

GPU_MON_PID=""

cleanup_monitor() {
    if [[ -n "$GPU_MON_PID" ]]; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        wait "$GPU_MON_PID" 2>/dev/null || true
    fi
    GPU_MON_PID=""
}
trap cleanup_monitor EXIT INT TERM

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

if [[ ! -f "$SUMMARY" ]]; then
    echo "model,N,decode_tokens_per_seq,status,wall_time_sec,max_kv_utilization_pct,total_pauses,output_dir" \
        > "$SUMMARY"
fi

# Point constants.py's DEFAULT_CONFIG_FILE at default.yml, commenting out
# whatever line is currently active. This is the only way to select a config
# file, since main.py takes no --config_file flag.
set_active_config() {
    local yml_name="$1"
    sed -i -E 's/^DEFAULT_CONFIG_FILE = /# DEFAULT_CONFIG_FILE = /' "$CONSTANTS_FILE"
    sed -i -E "s|^# DEFAULT_CONFIG_FILE = f\"\{ROOT_DIR\}/config/${yml_name}\"\$|DEFAULT_CONFIG_FILE = f\"{ROOT_DIR}/config/${yml_name}\"|" "$CONSTANTS_FILE"

    local active_count active_line
    active_count=$(grep -cE '^DEFAULT_CONFIG_FILE = ' "$CONSTANTS_FILE")
    active_line=$(grep -E '^DEFAULT_CONFIG_FILE = ' "$CONSTANTS_FILE")

    if [[ "$active_count" -ne 1 || "$active_line" != *"$yml_name"* ]]; then
        echo "[FATAL] Failed to set active config to $yml_name. Current active line(s):"
        echo "$active_line"
        exit 1
    fi
    echo "[CONFIG] Active config set to $yml_name"
}

# Prints "max_utilization_pct,total_pauses" for a run, reading the KV-block
# instrumentation and sequence_metrics.csv already added to this codebase.
extract_kv_and_preemption() {
    local kv_csv="$1"
    local seq_csv="$2"
    python3 - "$kv_csv" "$seq_csv" <<'PYEOF'
import sys, csv

kv_path, seq_path = sys.argv[1], sys.argv[2]

max_util = "NA"
try:
    with open(kv_path) as f:
        rows = list(csv.DictReader(f))
    if rows:
        max_util = max(float(r["utilization_pct"]) for r in rows)
except FileNotFoundError:
    pass

total_pauses = "NA"
try:
    with open(seq_path) as f:
        rows = list(csv.DictReader(f))
    if rows:
        total_pauses = sum(int(float(r.get("request_num_pauses", 0) or 0)) for r in rows)
except FileNotFoundError:
    pass

print(f"{max_util},{total_pauses}")
PYEOF
}

cd "$REPO" || exit 1
set_active_config "$CONFIG_YML"

echo "[GPU] Pinned to CUDA_VISIBLE_DEVICES=$GPU_ID (tensor_parallel_degree=1, pipeline_parallel_degree=1)"

# max_model_len must cover prefill(1) + decode(DECODE_TOKENS_PER_SEQ), and
# VLLMSchedulerConfig asserts max_model_len <= max_tokens_in_batch. Both are
# fixed regardless of N: vLLM's decode-continuation loop doesn't consume the
# prefill token budget, so max_tokens_in_batch only needs to cover
# max_model_len, not N.
# cap=$((DECODE_TOKENS_PER_SEQ + 64))

export RAY_ADDRESS=local

for model_pair in "${MODELS[@]}"; do
    model_name="${model_pair%%|*}"
    model_safe="${model_pair##*|}"

    for N in "${N_VALUES[@]}"; do
        cap=$((DECODE_TOKENS_PER_SEQ + 64))
        if (( N > cap )); then
            cap=$N
        fi

        RUN_NAME="max_decode_${model_safe}_n_${N}"
        FINAL_DIR="$OUTPUT_ROOT/$RUN_NAME"

        echo "# model=$model_safe N=$N decode_tokens=$DECODE_TOKENS_PER_SEQ output=$RUN_NAME" >> "$TIME_FILE"

        if [[ -d "$FINAL_DIR" ]]; then
            echo "[SKIP] Already exists: $FINAL_DIR"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "Running model=$model_safe N=$N decode_tokens=$DECODE_TOKENS_PER_SEQ (gpu=$GPU_ID, tp=1, pp=1)"
        echo "============================================================"

        MARKER=$(mktemp)
        LOG_TMP=$(mktemp)
        GPU_MON_TMP=$(mktemp)

        start_time=$(date +%s)

        nvidia-smi \
            -i "$GPU_ID" \
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
            --format=csv \
            -lms 250 > "$GPU_MON_TMP" 2>&1 &
        GPU_MON_PID=$!

        set +e
        python "$MAIN" \
            --model_name "$model_name" \
            --model_tensor_parallel_degree 1 \
            --model_pipeline_parallel_degree 1 \
            --model_max_model_len "$cap" \
            --synthetic_request_generator_length_provider fixed \
            --synthetic_request_generator_interval_provider static \
            --synthetic_request_generator_num_requests "$N" \
            --fixed_request_length_generator_prefill_tokens 1 \
            --fixed_request_length_generator_decode_tokens "$DECODE_TOKENS_PER_SEQ" \
            --replica_scheduler_provider vllm \
            --replica_scheduler_max_batch_size "$N" \
            --vllm_scheduler_max_tokens_in_batch "$cap" \
            --metrics_store_keep_individual_batch_metrics true \
            2>&1 | tee "$LOG_TMP"
        run_status=${PIPESTATUS[0]}
        set -e

        end_time=$(date +%s)
        wall_time=$((end_time - start_time))

        cleanup_monitor

        sleep 25 

        GENERATED_DIR=$(
            find "$OUTPUT_ROOT" \
                -mindepth 1 -maxdepth 1 -type d \
                -name '20??-??-??_*' \
                -newer "$MARKER" \
                -printf '%T@ %p\n' |
                sort -nr | head -n 1 | cut -d' ' -f2-
        )

        status="failed"
        if grep -qiE "CUDA out of memory|out of memory|OutOfMemoryError" "$LOG_TMP"; then
            status="oom"
        elif [[ "$run_status" -eq 0 && -n "$GENERATED_DIR" ]]; then
            status="success"
        fi

        kv_and_pauses="NA,NA"
        if [[ "$status" == "success" && -f "$GENERATED_DIR/replica_0/kv_block_usage.csv" ]]; then
            kv_and_pauses=$(extract_kv_and_preemption \
                "$GENERATED_DIR/replica_0/kv_block_usage.csv" \
                "$GENERATED_DIR/replica_0/sequence_metrics.csv")
        fi
        max_util="${kv_and_pauses%%,*}"
        total_pauses="${kv_and_pauses##*,}"

        if [[ -n "$GENERATED_DIR" && -d "$GENERATED_DIR" ]]; then
            mv "$LOG_TMP" "$GENERATED_DIR/run_log.log"
            mv "$GPU_MON_TMP" "$GENERATED_DIR/nvidia-smi-gpu${GPU_ID}.csv"
            if [[ "$status" == "success" ]]; then
                mv "$GENERATED_DIR" "$FINAL_DIR"
            else
                FINAL_DIR="${FINAL_DIR}_${status}_$(date +%Y%m%d_%H%M%S)"
                mv "$GENERATED_DIR" "$FINAL_DIR"
            fi
        else
            FINAL_DIR="${FINAL_DIR}_${status}_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$FINAL_DIR"
            mv "$LOG_TMP" "$FINAL_DIR/run_log.log"
            mv "$GPU_MON_TMP" "$FINAL_DIR/nvidia-smi-gpu${GPU_ID}.csv"
        fi

        echo "$model_safe,$N,$DECODE_TOKENS_PER_SEQ,$status,$wall_time,$max_util,$total_pauses,$FINAL_DIR" \
            >> "$SUMMARY"
        echo "[$status] model=$model_safe N=$N wall=${wall_time}s max_kv_util=${max_util}% pauses=${total_pauses}"

        rm -f "$MARKER"

        if [[ "$status" == "oom" ]]; then
            echo "[STOP] $model_safe hit OOM at N=$N — skipping larger N for this model."
            break
        fi
    done
done

echo ""
echo "Sweep complete. Summary: $SUMMARY"
