#!/usr/bin/env bash
set -uo pipefail

# chmod +x scripts/run_qwen_sweep.sh
# ./scripts/run_qwen_sweep.sh

# Use nohup so it continues after SSH/laptop disconnects:
# nohup ./scripts/run_qwen_sweep.sh \
#  > log/qwen_sweep.log 2>&1 &

# echo $! > log/qwen_sweep.pid
# tail -f log/qwen_sweep.log      Check progress:
# ps -fp "$(cat log/qwen_sweep.pid)"      ps -fp "$(cat log/qwen_sweep.pid)"
# kill "$(cat log/qwen_sweep.pid)"          kill "$(cat log/qwen_sweep.pid)"
# watch -n 10 'ls -lt benchmark_output/figure-1 | head'  To monitor output directories:
# tmux new -s qwen-sweep       Using tmux is even more convenient for reconnecting:
# ./scripts/run_qwen_sweep.sh
# Detach with Ctrl+B, then D. Reconnect later:
# tmux attach -t qwen-sweep

# pgrep -af "run_qwen_sweep.sh"

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
OUTPUT_ROOT="$REPO/benchmark_output/figure-1"
TIME_FILE="$REPO/log/time.txt"
LATEST_LOG="$REPO/log/sarathi_log.log"
SUMMARY="$OUTPUT_ROOT/arxiv_qwen_qps_chunk_sweep.csv"

# QPS_VALUES=(1 5 10 15 20)
QPS_VALUES=(10 15 20)
# QPS_VALUES=(20)
CHUNK_SIZES=(128 256 512 1024 2048 3072 4096)
# CHUNK_SIZES=(4096)

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
    echo "qps,chunk_size,status,inference_time_sec,wall_time_sec,output_dir" \
        > "$SUMMARY"
fi

cd "$REPO" || exit 1

for qps in "${QPS_VALUES[@]}"; do
    echo "" >> "$TIME_FILE"
    echo "# Poisson $qps request per seconds" >> "$TIME_FILE"

    for chunk_size in "${CHUNK_SIZES[@]}"; do
        RUN_NAME="arxiv_chunk_${chunk_size}_poison_${qps}_qps_qwen"
        FINAL_DIR="$OUTPUT_ROOT/$RUN_NAME"

        if [[ -d "$FINAL_DIR" ]]; then
            echo "[SKIP] Already exists: $FINAL_DIR"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "Running QPS=$qps, chunk_size=$chunk_size"
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
            --poisson_request_interval_generator_qps "$qps" \
            --sarathi_scheduler_chunk_size "$chunk_size" \
            2>&1 | tee "$LOG_TMP"
        run_status=${PIPESTATUS[0]}
        set -e

        end_time=$(date +%s)
        wall_time=$((end_time - start_time))

        cleanup_monitors

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

            echo "$qps,$chunk_size,failed,$inference_time,$wall_time,$FAILED_DIR" \
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

            echo "$qps,$chunk_size,success,$inference_time,$wall_time,$FINAL_DIR" \
                >> "$SUMMARY"

            echo "[SUCCESS] $FINAL_DIR"
        else
            FAILED_DIR="${FINAL_DIR}_failed_$(date +%Y%m%d_%H%M%S)"

            mv "$GPU_TMP" "$GENERATED_DIR/nvidia-smi-output.csv"
            mv "$TOP_TMP" "$GENERATED_DIR/top_output.txt"
            mv "$LOG_TMP" "$GENERATED_DIR/sarathi_log.log"
            mv "$GENERATED_DIR" "$FAILED_DIR"

            echo "$qps,$chunk_size,failed,$inference_time,$wall_time,$FAILED_DIR" \
                >> "$SUMMARY"

            rm -f "$MARKER"

            echo "[ERROR] Run failed. Output: $FAILED_DIR"
            exit "$run_status"
        fi

        rm -f "$MARKER"
    done
done

echo ""
echo "Sweep complete."
echo "Summary: $SUMMARY"