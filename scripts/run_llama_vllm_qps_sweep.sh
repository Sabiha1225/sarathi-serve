#!/usr/bin/env bash
set -uo pipefail

# chmod +x scripts/run_llama_vllm_qps_sweep.sh
# nohup ./scripts/run_llama_vllm_qps_sweep.sh \
#  > log/llama_vllm_qps_sweep.log 2>&1 &

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
OUTPUT_ROOT="$REPO/benchmark_output/figure-1"
TIME_FILE="$REPO/log/time.txt"
LATEST_LOG="$REPO/log/sarathi_log.log"

QPS_VALUES=(1 5 10 15 20)

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

trap cleanup_monitors EXIT INT TERM

mkdir -p "$OUTPUT_ROOT" "$REPO/log"
cd "$REPO" || exit 1

for qps in "${QPS_VALUES[@]}"; do
    RUN_NAME="arxiv_vllm_poison_${qps}_qps_llama"
    FINAL_DIR="$OUTPUT_ROOT/$RUN_NAME"

    if [[ -d "$FINAL_DIR" ]]; then
        echo "[SKIP] Already exists: $FINAL_DIR"
        continue
    fi

    echo
    echo "============================================================"
    echo "Running vLLM scheduler: QPS=$qps"
    echo "Output: $RUN_NAME"
    echo "============================================================"

    # benchmark_runner.py automatically appends inference time afterward.
    {
        echo
        echo "# vLLM Poisson $qps request per seconds"
    } >> "$TIME_FILE"

    # Used to identify the timestamped output created by this run.
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
        --poisson_request_interval_generator_qps "$qps" \
        2>&1 | tee "$LOG_TMP"

    RUN_STATUS=${PIPESTATUS[0]}

    set -e

    cleanup_monitors

    # Find the newest timestamped directory created by this run.
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

        echo "[ERROR] Generated benchmark directory was not found."
        exit 1
    fi

    # Place monitoring and console logs inside the benchmark directory.
    mv "$GPU_TMP" "$GENERATED_DIR/nvidia-smi-output.csv"
    mv "$TOP_TMP" "$GENERATED_DIR/top_output.txt"
    mv "$LOG_TMP" "$GENERATED_DIR/sarathi_log.log"

    {
        echo "scheduler=vllm"
        echo "qps=$qps"
        echo "dataset=arxiv"
        echo "model=llama"
    } > "$GENERATED_DIR/run_summary.txt"

    if [[ "$RUN_STATUS" -eq 0 ]]; then
        mv "$GENERATED_DIR" "$FINAL_DIR"
        cp "$FINAL_DIR/sarathi_log.log" "$LATEST_LOG"

        echo "[SUCCESS] Saved as: $FINAL_DIR"
    else
        FAILED_DIR="${FINAL_DIR}_failed_$(date +%Y%m%d_%H%M%S)"
        mv "$GENERATED_DIR" "$FAILED_DIR"

        echo "[ERROR] Run failed: $FAILED_DIR"
        exit "$RUN_STATUS"
    fi
done

echo
echo "vLLM QPS sweep completed."