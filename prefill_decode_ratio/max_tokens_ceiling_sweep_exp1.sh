#!/usr/bin/env bash
set -uo pipefail

# nvidia-smi-gpu0.csv, run_log.log, batch_metrics.csv

# chmod +x prefill_decode_ratio/max_tokens_ceiling_sweep_exp1.sh
# nohup ./prefill_decode_ratio/max_tokens_ceiling_sweep_exp1.sh \
#   > log/max_tokens_ceiling_sweep_exp1.log 2>&1 &

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
CONSTANTS_FILE="$REPO/sarathi/benchmark/constants.py"
OUTPUT_ROOT="$REPO/benchmark_output/pd_ratio_max_tokens_ceiling_exp1"
TIME_FILE="$REPO/log/pd_ratio_max_tokens_ceiling_exp1.txt"
LOG_DIR="$REPO/log"
SUMMARY="$OUTPUT_ROOT/max_tokens_ceiling_summary.csv"
CONFIG_YML="default.yml"

# Pin to one GPU. This machine has two L40S; change to 1 to use the other.
GPU_ID=0
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Doubling sweep. The script stops early for a model once it OOMs.
P_VALUES=(512 1024 2048 4096 8192 16384 24576 32768)

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
    echo "model,P,status,wall_time_sec,batch_execution_time_sec,output_dir" > "$SUMMARY"
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

# Prints the batch_execution_time of the prefill iteration (the row with the
# largest batch_num_prefill_tokens) from a run's batch_metrics.csv.
extract_batch_execution_time() {
    local csv_path="$1"
    python3 - "$csv_path" <<'PYEOF'
import sys, csv
path = sys.argv[1]
try:
    with open(path) as f:
        rows = list(csv.DictReader(f))
except FileNotFoundError:
    print("")
    sys.exit(0)
prefill_rows = [r for r in rows if float(r.get("batch_num_prefill_tokens", 0)) > 0]
if not prefill_rows:
    print("")
    sys.exit(0)
best = max(prefill_rows, key=lambda r: float(r["batch_num_prefill_tokens"]))
print(best["batch_execution_time"])
PYEOF
}

cd "$REPO" || exit 1
set_active_config "$CONFIG_YML"

echo "[GPU] Pinned to CUDA_VISIBLE_DEVICES=$GPU_ID (tensor_parallel_degree=1, pipeline_parallel_degree=1)"

for model_pair in "${MODELS[@]}"; do
    model_name="${model_pair%%|*}"
    model_safe="${model_pair##*|}"

    for P in "${P_VALUES[@]}"; do
        RUN_NAME="max_tokens_${model_safe}_p_${P}"
        FINAL_DIR="$OUTPUT_ROOT/$RUN_NAME"
        echo "# model=$model_safe P=$P output=$RUN_NAME" >> "$TIME_FILE"
        if [[ -d "$FINAL_DIR" ]]; then
            echo "[SKIP] Already exists: $FINAL_DIR"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "Running model=$model_safe P=$P (gpu=$GPU_ID, tp=1, pp=1)"
        echo "============================================================"

        MARKER=$(mktemp)
        LOG_TMP=$(mktemp)
        GPU_MON_TMP=$(mktemp)

        # max_model_len must be >= P+1 (prompt + 1 decode token), and
        # VLLMSchedulerConfig asserts max_model_len <= max_tokens_in_batch,
        # so both are set to the same padded value.
        cap=$((P + 64))

        start_time=$(date +%s)

        nvidia-smi \
            -i "$GPU_ID" \
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
            --format=csv \
            -lms 100 > "$GPU_MON_TMP" 2>&1 &
        GPU_MON_PID=$!

        set +e
        python "$MAIN" \
            --model_name "$model_name" \
            --model_tensor_parallel_degree 1 \
            --model_pipeline_parallel_degree 1 \
            --model_max_model_len "$cap" \
            --synthetic_request_generator_length_provider fixed \
            --synthetic_request_generator_interval_provider static \
            --synthetic_request_generator_num_requests 1 \
            --fixed_request_length_generator_prefill_tokens "$P" \
            --fixed_request_length_generator_decode_tokens 1 \
            --replica_scheduler_provider vllm \
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

        batch_time="NA"
        if [[ "$status" == "success" && -f "$GENERATED_DIR/replica_0/batch_metrics.csv" ]]; then
            extracted=$(extract_batch_execution_time "$GENERATED_DIR/replica_0/batch_metrics.csv")
            [[ -n "$extracted" ]] && batch_time="$extracted"
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

        echo "$model_safe,$P,$status,$wall_time,$batch_time,$FINAL_DIR" >> "$SUMMARY"
        echo "[$status] model=$model_safe P=$P wall=${wall_time}s batch_execution_time=${batch_time}"

        rm -f "$MARKER"

        if [[ "$status" == "oom" ]]; then
            echo "[STOP] $model_safe hit OOM at P=$P — skipping larger P for this model."
            break
        fi
    done
done

echo ""
echo "Sweep complete. Summary: $SUMMARY"
