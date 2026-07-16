#!/usr/bin/env bash
set -uo pipefail
# chmod +x scripts/run_llama_gpu_preemption_sweep.sh
# nohup ./scripts/run_llama_gpu_preemption_sweep.sh \
#  > log/llama_gpu_preemption_sweep.log 2>&1 &

# echo $! > log/llama_gpu_preemption_sweep.pid

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
OUTPUT_ROOT="$REPO/benchmark_output/figure-1"
TIME_FILE="$REPO/log/time.txt"
LATEST_LOG="$REPO/log/sarathi_log.log"

GPU_VALUES=(0.85 0.65 0.45 0.25)
QPS=200
CHUNK_SIZE=1024

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

trap cleanup_monitors EXIT INT TERM

mkdir -p "$OUTPUT_ROOT" "$REPO/log"
cd "$REPO" || exit 1

for gpu_util in "${GPU_VALUES[@]}"; do
    # Convert 0.85 -> 8_5 and 0.25 -> 2_5.
    gpu_label=$(
        awk -v value="$gpu_util" 'BEGIN { printf "%g", value * 10 }' |
            tr '.' '_'
    )

    # Convert 0.85 -> .85 for time.txt.
    display_gpu="${gpu_util#0}"

    RUN_NAME="arxiv_sarathi_gpu_${gpu_label}_llama"
    FINAL_DIR="$OUTPUT_ROOT/$RUN_NAME"

    if [[ -d "$FINAL_DIR" ]]; then
        echo "[SKIP] Already exists: $FINAL_DIR"
        continue
    fi

    echo
    echo "============================================================"
    echo "GPU=$gpu_util, QPS=$QPS, chunk_size=$CHUNK_SIZE"
    echo "Output=$RUN_NAME"
    echo "============================================================"

    # benchmark_runner.py will append inference time after this line.
    {
        echo
        echo "Sarathi_serve : 200 Requests - gpu = $display_gpu , 0 Preemption, chunk size $CHUNK_SIZE"
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
        --gpu_memory_utilization "$gpu_util" \
        --poisson_request_interval_generator_qps "$QPS" \
        --sarathi_scheduler_chunk_size "$CHUNK_SIZE" \
        2>&1 | tee "$LOG_TMP"

    RUN_STATUS=${PIPESTATUS[0]}

    set -e

    cleanup_monitors

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

echo
echo "GPU utilization sweep completed."