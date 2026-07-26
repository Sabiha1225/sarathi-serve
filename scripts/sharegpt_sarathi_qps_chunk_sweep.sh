#!/usr/bin/env bash
set -uo pipefail

# chmod +x scripts/sharegpt_sarathi_qps_chunk_sweep.sh
# ./scripts/sharegpt_sarathi_qps_chunk_sweep.sh

# nohup ./scripts/sharegpt_sarathi_qps_chunk_sweep.sh \
#  > log/sharegpt_sarathi_qps_chunk_sweep.log 2>&1 &


REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
CONSTANTS_FILE="$REPO/sarathi/benchmark/constants.py"
OUTPUT_ROOT="$REPO/benchmark_output/sharegpt"
TIME_FILE="$REPO/log/time_sharegpt.txt"
LATEST_LOG="$REPO/log/sarathi_log.log"
SUMMARY="$OUTPUT_ROOT/sharegpt_qps_chunk_sweep_summary.csv"
CONFIG_YML="sharegpt_sarathi_varied_chunk_size_varried_req_time.yml"

QPS_VALUES=(1 5 10 15 20)
CHUNK_SIZES=(128 256 512 1024 2048 3072 4096)

# model_name|safe_label
MODELS=(
    "meta-llama/Llama-2-7b-hf|llama"
    "mistralai/Mistral-7B-Instruct-v0.3|mistral"
    "DiscoResearch/mixtral-7b-8expert|mixtral"
    "Qwen/Qwen-7B|qwen"
    "01-ai/Yi-6B|yi"
)

GPU_PID=""
TOP_PID=""

cleanup_monitors() {
    if [[ -n "$GPU_PID" ]]; then
        kill "$GPU_PID" 2>/dev/null || true
        wait "$GPU_PID" 2>/dev/null || true
    fi

    if [[ -n "$TOP_PID" ]]; then
        kill "$TOP_PID" 2>/dev/null || true
        wait "$TOP_PID" 2>/dev/null || true
    fi

    GPU_PID=""
    TOP_PID=""
}

trap cleanup_monitors EXIT INT TERM

mkdir -p "$OUTPUT_ROOT" "$REPO/log"

if [[ ! -f "$SUMMARY" ]]; then
    echo "model,qps,chunk_size,status,inference_time_sec,wall_time_sec,output_dir" \
        > "$SUMMARY"
fi

# Point constants.py's DEFAULT_CONFIG_FILE at the sharegpt yml, commenting out
# whatever line is currently active. This is the only way to select a config
# file, since main.py takes no --config_file flag. Model switching itself is
# done purely via --model_name below, so this only needs to run once.
set_active_config() {
    local yml_name="$1"

    sed -i -E 's/^DEFAULT_CONFIG_FILE = /# DEFAULT_CONFIG_FILE = /' "$CONSTANTS_FILE"
    sed -i -E "s|^# DEFAULT_CONFIG_FILE = f\"\{ROOT_DIR\}/config/${yml_name}\"\$|DEFAULT_CONFIG_FILE = f\"{ROOT_DIR}/config/${yml_name}\"|" "$CONSTANTS_FILE"

    local active_count
    active_count=$(grep -cE '^DEFAULT_CONFIG_FILE = ' "$CONSTANTS_FILE")
    local active_line
    active_line=$(grep -E '^DEFAULT_CONFIG_FILE = ' "$CONSTANTS_FILE")

    if [[ "$active_count" -ne 1 || "$active_line" != *"$yml_name"* ]]; then
        echo "[FATAL] Failed to set active config to $yml_name. Current active line(s):"
        echo "$active_line"
        exit 1
    fi
    echo "[CONFIG] Active config set to $yml_name"
}

cd "$REPO" || exit 1

set_active_config "$CONFIG_YML"

for model_pair in "${MODELS[@]}"; do
    model_name="${model_pair%%|*}"
    model_safe="${model_pair##*|}"

    for qps in "${QPS_VALUES[@]}"; do
        echo "" >> "$TIME_FILE"
        echo "# model=$model_safe Poisson $qps request per seconds" >> "$TIME_FILE"

        for chunk_size in "${CHUNK_SIZES[@]}"; do
            RUN_NAME="sharegpt_chunk_${chunk_size}_poison_${qps}_qps_${model_safe}"
            FINAL_DIR="$OUTPUT_ROOT/$RUN_NAME"

            if [[ -d "$FINAL_DIR" ]]; then
                echo "[SKIP] Already exists: $FINAL_DIR"
                continue
            fi

            echo ""
            echo "============================================================"
            echo "Running model=$model_safe QPS=$qps, chunk_size=$chunk_size"
            echo "============================================================"

            echo "# chunk_size=$chunk_size output=$RUN_NAME" >> "$TIME_FILE"

            MARKER=$(mktemp)
            GPU_TMP=$(mktemp)
            TOP_TMP=$(mktemp)
            LOG_TMP=$(mktemp)

            start_time=$(date +%s)

            nvidia-smi \
                --query-gpu=timestamp,index,memory.used,utilization.gpu \
                --format=csv \
                -lms 250 > "$GPU_TMP" 2>&1 &
            GPU_PID=$!

            top \
                -d 4.0 \
                -u "$(whoami)" \
                -b > "$TOP_TMP" 2>&1 &
            TOP_PID=$!

            set +e
            python "$MAIN" \
                --model_name "$model_name" \
                --poisson_request_interval_generator_qps "$qps" \
                --sarathi_scheduler_chunk_size "$chunk_size" \
                2>&1 | tee "$LOG_TMP"
            run_status=${PIPESTATUS[0]}
            set -e

            end_time=$(date +%s)
            wall_time=$((end_time - start_time))

            cleanup_monitors

            sleep 25

            GENERATED_DIR=$(
                find "$OUTPUT_ROOT" \
                    -mindepth 1 \
                    -maxdepth 1 \
                    -type d \
                    -name '20??-??-??_*' \
                    -newer "$MARKER" \
                    -printf '%T@ %p\n' |
                    sort -nr |
                    head -n 1 |
                    cut -d' ' -f2-
            )

            inference_time=$(
                grep -oE 'Total time taken: [0-9.]+ seconds' "$LOG_TMP" |
                    tail -n 1 |
                    awk '{print $4}'
            )

            if [[ -z "$inference_time" ]]; then
                inference_time="NA"
            fi

            if [[ -z "$GENERATED_DIR" || ! -d "$GENERATED_DIR" ]]; then
                FAILED_DIR="${FINAL_DIR}_failed_$(date +%Y%m%d_%H%M%S)"
                mkdir -p "$FAILED_DIR"

                mv "$GPU_TMP" "$FAILED_DIR/nvidia-smi-output.csv"
                mv "$TOP_TMP" "$FAILED_DIR/top_output.txt"
                mv "$LOG_TMP" "$FAILED_DIR/sarathi_log.log"

                echo "$model_safe,$qps,$chunk_size,failed,$inference_time,$wall_time,$FAILED_DIR" \
                    >> "$SUMMARY"

                rm -f "$MARKER"

                echo "[ERROR] Benchmark output directory was not found."
                exit 1
            fi

            if [[ "$run_status" -eq 0 ]]; then
                mv "$GPU_TMP" "$GENERATED_DIR/nvidia-smi-output.csv"
                mv "$TOP_TMP" "$GENERATED_DIR/top_output.txt"
                mv "$LOG_TMP" "$GENERATED_DIR/sarathi_log.log"

                mv "$GENERATED_DIR" "$FINAL_DIR"
                cp "$FINAL_DIR/sarathi_log.log" "$LATEST_LOG"

                echo "$model_safe,$qps,$chunk_size,success,$inference_time,$wall_time,$FINAL_DIR" \
                    >> "$SUMMARY"

                echo "[SUCCESS] $FINAL_DIR"
            else
                FAILED_DIR="${FINAL_DIR}_failed_$(date +%Y%m%d_%H%M%S)"

                mv "$GPU_TMP" "$GENERATED_DIR/nvidia-smi-output.csv"
                mv "$TOP_TMP" "$GENERATED_DIR/top_output.txt"
                mv "$LOG_TMP" "$GENERATED_DIR/sarathi_log.log"
                mv "$GENERATED_DIR" "$FAILED_DIR"

                echo "$model_safe,$qps,$chunk_size,failed,$inference_time,$wall_time,$FAILED_DIR" \
                    >> "$SUMMARY"

                rm -f "$MARKER"

                echo "[ERROR] Run failed. Output: $FAILED_DIR"
                exit "$run_status"
            fi

            rm -f "$MARKER"
        done
    done
done

echo ""
echo "Sweep complete."
echo "Summary: $SUMMARY"
