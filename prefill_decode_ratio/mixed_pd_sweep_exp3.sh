#!/usr/bin/env bash
set -uo pipefail

# chmod +x prefill_decode_ratio/mixed_pd_sweep_exp3.sh
# nohup ./prefill_decode_ratio/mixed_pd_sweep_exp3.sh \
#   > log/mixed_pd_sweep_exp3.log 2>&1 &

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
CONSTANTS_FILE="$REPO/sarathi/benchmark/constants.py"
OUTPUT_ROOT="$REPO/benchmark_output/pd_ratio_mixed_sweep_exp3"
LOG_DIR="$REPO/log"
SUMMARY="$OUTPUT_ROOT/mixed_sweep_summary.csv"
TIME_FILE="$REPO/log/pd_ratio_mixed_sweep_exp3.txt"
CONFIG_YML="default.yml"

# Pin to one GPU. This machine has two L40S; change to 1 to use the other.
GPU_ID=0
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export RAY_ADDRESS=local

# Decode tokens per sequence — shorter than Experiment 2's 2048, since P
# grows up to 3584 here and model_max_model_len = P + decode_tokens must
# stay under Llama-2/Yi's native 4096.
DECODE_TOKENS_PER_SEQ=256

# Prefill tokens per request — the axis actually being studied. Capped at
# 3584 so P + DECODE_TOKENS_PER_SEQ + slack stays under 4096.
P_VALUES=(1 128 256 512 1024 2048 3584)

# Sarathi chunk sizes to sweep, same range as P.
CHUNK_SIZES=(128 256 512 1024 2048 3584)

# model_name|safe_label|N  (N = decode knee measured in Experiment 2)
MODELS=(
    "meta-llama/Llama-2-7b-hf|llama|32"
    "mistralai/Mistral-7B-Instruct-v0.3|mistral|128"
    "DiscoResearch/mixtral-7b-8expert|mixtral|2048"
    "Qwen/Qwen-7B|qwen|32"
    "01-ai/Yi-6B|yi|256"
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
    echo "model,N,P,decode_tokens,scheduler,chunk_size,status,wall_time_sec,mean_ttft_sec,p99_ttft_sec,mean_tpot_sec,p99_tpot_sec,output_dir" \
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

# Prints "mean_ttft,p99_ttft,mean_tpot,p99_tpot" from a run's
# sequence_metrics.csv. TTFT = prefill_e2e_time, TPOT =
# decode_time_execution_plus_preemption_normalized.
extract_ttft_tpot() {
    local csv_path="$1"
    python3 - "$csv_path" <<'PYEOF'
import sys, csv

path = sys.argv[1]
try:
    with open(path) as f:
        rows = list(csv.DictReader(f))
except FileNotFoundError:
    print("NA,NA,NA,NA")
    sys.exit(0)


def stats(col):
    vals = []
    for r in rows:
        v = r.get(col, "")
        if v in (None, "", "None"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    if not vals:
        return "NA", "NA"
    vals.sort()
    mean = sum(vals) / len(vals)
    p99_idx = min(len(vals) - 1, int(len(vals) * 0.99))
    return f"{mean:.6f}", f"{vals[p99_idx]:.6f}"


mean_ttft, p99_ttft = stats("prefill_e2e_time")
mean_tpot, p99_tpot = stats("decode_time_execution_plus_preemption_normalized")
print(f"{mean_ttft},{p99_ttft},{mean_tpot},{p99_tpot}")
PYEOF
}

cd "$REPO" || exit 1
set_active_config "$CONFIG_YML"

echo "[GPU] Pinned to CUDA_VISIBLE_DEVICES=$GPU_ID (tensor_parallel_degree=1, pipeline_parallel_degree=1)"

# Runs one (model, N, P, scheduler, chunk_size) combination end to end:
# launches python, monitors the GPU, classifies the outcome, extracts
# TTFT/TPOT, and appends one row to $SUMMARY plus one label to $TIME_FILE.
run_one() {
    local model_name="$1" model_safe="$2" N="$3" P="$4" scheduler="$5"
    local chunk_size="$6" run_name="$7"

    local FINAL_DIR="$OUTPUT_ROOT/$run_name"
    if [[ -d "$FINAL_DIR" ]]; then
        echo "[SKIP] Already exists: $FINAL_DIR"
        return
    fi

    echo "# model=$model_safe N=$N P=$P decode_tokens=$DECODE_TOKENS_PER_SEQ scheduler=$scheduler chunk_size=$chunk_size output=$run_name" \
        >> "$TIME_FILE"

    echo ""
    echo "============================================================"
    echo "Running model=$model_safe N=$N P=$P scheduler=$scheduler chunk_size=$chunk_size"
    echo "============================================================"

    local MARKER LOG_TMP GPU_MON_TMP cap start_time end_time wall_time
    MARKER=$(mktemp)
    LOG_TMP=$(mktemp)
    GPU_MON_TMP=$(mktemp)

    # model_max_model_len must cover P + decode_tokens (+ slack), and for
    # vLLM, VLLMSchedulerConfig asserts max_model_len <= max_tokens_in_batch,
    # so both share this same value.
    cap=$((P + DECODE_TOKENS_PER_SEQ + 64))
    if (( N > cap )); then
        cap=$N
    fi

    start_time=$(date +%s)

    nvidia-smi \
        -i "$GPU_ID" \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv \
        -lms 250 > "$GPU_MON_TMP" 2>&1 &
    GPU_MON_PID=$!

    set +e
    if [[ "$scheduler" == "vllm" ]]; then
        python "$MAIN" \
            --model_name "$model_name" \
            --model_tensor_parallel_degree 1 \
            --model_pipeline_parallel_degree 1 \
            --model_max_model_len "$cap" \
            --synthetic_request_generator_length_provider fixed \
            --synthetic_request_generator_interval_provider static \
            --synthetic_request_generator_num_requests "$N" \
            --fixed_request_length_generator_prefill_tokens "$P" \
            --fixed_request_length_generator_decode_tokens "$DECODE_TOKENS_PER_SEQ" \
            --replica_scheduler_provider vllm \
            --replica_scheduler_max_batch_size "$N" \
            --vllm_scheduler_max_tokens_in_batch "$cap" \
            --metrics_store_keep_individual_batch_metrics true \
            2>&1 | tee "$LOG_TMP"
    else
        python "$MAIN" \
            --model_name "$model_name" \
            --model_tensor_parallel_degree 1 \
            --model_pipeline_parallel_degree 1 \
            --model_max_model_len "$cap" \
            --synthetic_request_generator_length_provider fixed \
            --synthetic_request_generator_interval_provider static \
            --synthetic_request_generator_num_requests "$N" \
            --fixed_request_length_generator_prefill_tokens "$P" \
            --fixed_request_length_generator_decode_tokens "$DECODE_TOKENS_PER_SEQ" \
            --replica_scheduler_provider sarathi \
            --replica_scheduler_max_batch_size "$N" \
            --sarathi_scheduler_chunk_size "$chunk_size" \
            --sarathi_scheduler_enable_dynamic_chunking_schedule false \
            --metrics_store_keep_individual_batch_metrics true \
            2>&1 | tee "$LOG_TMP"
    fi
    local run_status=${PIPESTATUS[0]}
    set -e

    end_time=$(date +%s)
    wall_time=$((end_time - start_time))

    cleanup_monitor

    local GENERATED_DIR
    GENERATED_DIR=$(
        find "$OUTPUT_ROOT" \
            -mindepth 1 -maxdepth 1 -type d \
            -name '20??-??-??_*' \
            -newer "$MARKER" \
            -printf '%T@ %p\n' |
            sort -nr | head -n 1 | cut -d' ' -f2-
    )

    local status="failed"
    if grep -qiE "CUDA out of memory|out of memory|OutOfMemoryError" "$LOG_TMP"; then
        status="oom"
    elif [[ "$run_status" -eq 0 && -n "$GENERATED_DIR" ]]; then
        status="success"
    fi

    local metrics="NA,NA,NA,NA"
    if [[ -n "$GENERATED_DIR" && -f "$GENERATED_DIR/replica_0/sequence_metrics.csv" ]]; then
        metrics=$(extract_ttft_tpot "$GENERATED_DIR/replica_0/sequence_metrics.csv")
    fi

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

    local chunk_label="$chunk_size"
    [[ "$scheduler" == "vllm" ]] && chunk_label="NA"

    echo "$model_safe,$N,$P,$DECODE_TOKENS_PER_SEQ,$scheduler,$chunk_label,$status,$wall_time,$metrics,$FINAL_DIR" \
        >> "$SUMMARY"
    echo "[$status] model=$model_safe N=$N P=$P scheduler=$scheduler chunk_size=$chunk_label wall=${wall_time}s"

    rm -f "$MARKER"
}

for model_pair in "${MODELS[@]}"; do
    model_name="${model_pair%%|*}"
    rest="${model_pair#*|}"
    model_safe="${rest%%|*}"
    N="${rest##*|}"

    # vLLM baseline: sweep P only, no chunking involved.
    for P in "${P_VALUES[@]}"; do
        run_one "$model_name" "$model_safe" "$N" "$P" "vllm" "0" \
            "pdmix_${model_safe}_vllm_p${P}"
    done

    # Sarathi: full P x chunk_size grid.
    for chunk_size in "${CHUNK_SIZES[@]}"; do
        for P in "${P_VALUES[@]}"; do
            run_one "$model_name" "$model_safe" "$N" "$P" "sarathi" "$chunk_size" \
                "pdmix_${model_safe}_sarathi_c${chunk_size}_p${P}"
        done
    done
done

echo ""
echo "Sweep complete. Summary: $SUMMARY"
