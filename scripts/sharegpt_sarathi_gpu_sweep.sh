#!/usr/bin/env bash
set -uo pipefail
# chmod +x scripts/sharegpt_sarathi_gpu_sweep.sh
# nohup ./scripts/sharegpt_sarathi_gpu_sweep.sh \
#  > log/sharegpt_sarathi_gpu_sweep.log 2>&1 &

# echo $! > log/sharegpt_sarathi_gpu_sweep.pid

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
CONSTANTS_FILE="$REPO/sarathi/benchmark/constants.py"
OUTPUT_ROOT="$REPO/benchmark_output/sharegpt"
TIME_FILE="$REPO/log/time_sharegpt.txt"
LATEST_LOG="$REPO/log/sarathi_log.log"
CONFIG_YML="sharegpt_sarathi_varied_chunk_size_varried_req_time.yml"

GPU_VALUES=(0.85 0.65 0.45 0.25)
QPS=20
CHUNK_SIZE=1024

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
    for pid in "$GPU_PID" "$TOP_PID"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done

    GPU_PID=""
    TOP_PID=""
}

count_waiting_preemptions() {
    local csv_file="$1"

    python - "$csv_file" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)

if "state" not in df.columns:
    print(0)
else:
    states = df["state"].astype(str).str.strip().str.upper()
    print(int((states == "WAITING").sum()))
PY
}

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

trap cleanup_monitors EXIT INT TERM

mkdir -p "$OUTPUT_ROOT" "$REPO/log"
cd "$REPO" || exit 1

set_active_config "$CONFIG_YML"

for model_pair in "${MODELS[@]}"; do
    model_name="${model_pair%%|*}"
    model_safe="${model_pair##*|}"

    for gpu_util in "${GPU_VALUES[@]}"; do
        # Convert 0.85 -> 8_5 and 0.25 -> 2_5.
        gpu_label=$(
            awk -v value="$gpu_util" 'BEGIN { printf "%g", value * 10 }' |
                tr '.' '_'
        )

        # Convert 0.85 -> .85 for time_sharegpt.txt.
        display_gpu="${gpu_util#0}"

        RUN_NAME="sharegpt_sarathi_gpu_${gpu_label}_${model_safe}"
        FINAL_DIR="$OUTPUT_ROOT/$RUN_NAME"

        if [[ -d "$FINAL_DIR" ]]; then
            echo "[SKIP] Already exists: $FINAL_DIR"
            continue
        fi

        echo
        echo "============================================================"
        echo "model=$model_safe GPU=$gpu_util, QPS=$QPS, chunk_size=$CHUNK_SIZE"
        echo "Output=$RUN_NAME"
        echo "============================================================"

        # benchmark_runner.py will append inference time after this line.
        {
            echo
            echo "Sarathi_serve sharegpt : model=$model_safe 200 Requests - gpu = $display_gpu , 0 Preemption, chunk size $CHUNK_SIZE"
        } >> "$TIME_FILE"

        MARKER=$(mktemp)
        GPU_TMP=$(mktemp)
        TOP_TMP=$(mktemp)
        LOG_TMP=$(mktemp)

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
            --gpu_memory_utilization "$gpu_util" \
            --poisson_request_interval_generator_qps "$QPS" \
            --sarathi_scheduler_chunk_size "$CHUNK_SIZE" \
            2>&1 | tee "$LOG_TMP"

        RUN_STATUS=${PIPESTATUS[0]}

        set -e

        cleanup_monitors

        sleep 25

        # Find the timestamped directory created after this run started.
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

        rm -f "$MARKER"

        if [[ -z "$GENERATED_DIR" || ! -d "$GENERATED_DIR" ]]; then
            FAILED_DIR="${FINAL_DIR}_failed_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$FAILED_DIR"

            mv "$GPU_TMP" "$FAILED_DIR/nvidia-smi-output.csv"
            mv "$TOP_TMP" "$FAILED_DIR/top_output.txt"
            mv "$LOG_TMP" "$FAILED_DIR/sarathi_log.log"

            echo "preemption count = unavailable" >> "$TIME_FILE"
            echo "[ERROR] Generated benchmark directory was not found."
            exit 1
        fi

        PREEMPTION_FILE="$GENERATED_DIR/replica_0/preemption_details.csv"

        if [[ -f "$PREEMPTION_FILE" ]]; then
            PREEMPTION_COUNT=$(count_waiting_preemptions "$PREEMPTION_FILE")
        else
            PREEMPTION_COUNT=0
            echo "[WARNING] Missing: $PREEMPTION_FILE"
        fi

        echo "preemption count = $PREEMPTION_COUNT" >> "$TIME_FILE"

        mv "$GPU_TMP" "$GENERATED_DIR/nvidia-smi-output.csv"
        mv "$TOP_TMP" "$GENERATED_DIR/top_output.txt"
        mv "$LOG_TMP" "$GENERATED_DIR/sarathi_log.log"

        # Save a small run summary inside the result directory.
        {
            echo "model=$model_safe"
            echo "gpu_memory_utilization=$gpu_util"
            echo "qps=$QPS"
            echo "chunk_size=$CHUNK_SIZE"
            echo "waiting_preemption_count=$PREEMPTION_COUNT"
        } > "$GENERATED_DIR/run_summary.txt"

        if [[ "$RUN_STATUS" -eq 0 ]]; then
            mv "$GENERATED_DIR" "$FINAL_DIR"
            cp "$FINAL_DIR/sarathi_log.log" "$LATEST_LOG"

            echo "[SUCCESS] Saved as: $FINAL_DIR"
            echo "[SUCCESS] WAITING preemptions: $PREEMPTION_COUNT"
        else
            FAILED_DIR="${FINAL_DIR}_failed_$(date +%Y%m%d_%H%M%S)"
            mv "$GENERATED_DIR" "$FAILED_DIR"

            echo "[ERROR] Run failed: $FAILED_DIR"
            echo "[ERROR] WAITING preemptions before failure: $PREEMPTION_COUNT"
            exit "$RUN_STATUS"
        fi
    done
done

echo
echo "GPU utilization sweep completed for all models."
