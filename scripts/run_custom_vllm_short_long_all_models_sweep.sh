#!/usr/bin/env bash
set -uo pipefail

# chmod +x scripts/run_custom_vllm_short_long_all_models_sweep.sh
# nohup ./scripts/run_custom_vllm_short_long_all_models_sweep.sh > log/custom_vllm_short_long_all_models_sweep.log 2>&1 &
# echo $! > log/custom_vllm_short_long_all_models_sweep.pid
# tail -f log/custom_vllm_short_long_all_models_sweep.log

REPO="/home/sabiha/sarathi-serve"
MAIN="$REPO/sarathi/benchmark/main.py"
OUTPUT_ROOT="$REPO/benchmark_output/figure-1"
TIME_FILE="$REPO/log/time.txt"
LATEST_LOG="$REPO/log/sarathi_log.log"
SUMMARY="$OUTPUT_ROOT/custom_vllm_short_long_all_models_sweep_summary.csv"

DATASET_LABEL="custom_vllm_short_input_long_output"

MODELS=(
    "meta-llama/Llama-2-7b-hf|llama2_7b"
    "mistralai/Mistral-7B-Instruct-v0.3|mistral_7b"
    "DiscoResearch/mixtral-7b-8expert|mixtral_7b_8expert"
    "Qwen/Qwen-7B|qwen_7b"
    "01-ai/Yi-6B|yi_6b"
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
    echo "sweep_type,model,num_requests,qps,status,inference_time_sec,wall_time_sec,output_dir" > "$SUMMARY"
fi

run_one() {
    local sweep_type="$1"
    local model_name="$2"
    local model_safe="$3"
    local num_requests="$4"
    local qps="$5"

    local qps_safe="${qps/./_}"
    local run_name="${DATASET_LABEL}_${sweep_type}_num_requests_${num_requests}_qps_${qps_safe}_${model_safe}"
    local final_dir="$OUTPUT_ROOT/$run_name"

    if [[ -d "$final_dir" ]]; then
        echo "[SKIP] Already exists: $final_dir"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "Running $sweep_type model=$model_name num_requests=$num_requests qps=$qps"
    echo "Output: $run_name"
    echo "============================================================"

    echo "" >> "$TIME_FILE"
    echo "Custom vllm short-long $sweep_type: model=$model_name num_requests=$num_requests qps=$qps" >> "$TIME_FILE"

    local marker
    local gpu_tmp
    local top_tmp
    local log_tmp

    marker="$(mktemp)"
    gpu_tmp="$(mktemp)"
    top_tmp="$(mktemp)"
    log_tmp="$(mktemp)"

    local start_time
    start_time="$(date +%s)"

    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,utilization.gpu \
        --format=csv \
        -lms 250 > "$gpu_tmp" 2>&1 &
    GPU_PID=$!

    top \
        -d 4.0 \
        -u "$(whoami)" \
        -b > "$top_tmp" 2>&1 &
    TOP_PID=$!

    set +e
    python "$MAIN" \
        --model_name "$model_name" \
        --synthetic_request_generator_num_requests "$num_requests" \
        --poisson_request_interval_generator_qps "$qps" \
        2>&1 | tee "$log_tmp"
    local run_status="${PIPESTATUS[0]}"
    set -e

    local end_time
    local wall_time
    end_time="$(date +%s)"
    wall_time="$((end_time - start_time))"

    cleanup_monitors

    local generated_dir
    generated_dir=$(
        find "$OUTPUT_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name '20??-??-??_*' \
            -newer "$marker" \
            -printf '%T@ %p\n' |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
    )

    local inference_time
    inference_time=$(
        grep -oE 'Total time taken: [0-9.]+ seconds' "$log_tmp" |
            tail -n 1 |
            awk '{print $4}'
    )

    if [[ -z "$inference_time" ]]; then
        inference_time="NA"
    fi

    if [[ -z "$generated_dir" || ! -d "$generated_dir" ]]; then
        local failed_dir="${final_dir}_failed_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$failed_dir"

        mv "$gpu_tmp" "$failed_dir/nvidia-smi-output.csv"
        mv "$top_tmp" "$failed_dir/top_output.txt"
        mv "$log_tmp" "$failed_dir/sarathi_log.log"

        echo "$sweep_type,$model_safe,$num_requests,$qps,failed,$inference_time,$wall_time,$failed_dir" >> "$SUMMARY"

        rm -f "$marker"

        echo "[ERROR] Benchmark output directory was not found."
        exit 1
    fi

    mv "$gpu_tmp" "$generated_dir/nvidia-smi-output.csv"
    mv "$top_tmp" "$generated_dir/top_output.txt"
    mv "$log_tmp" "$generated_dir/sarathi_log.log"

    if [[ "$run_status" -eq 0 ]]; then
        mv "$generated_dir" "$final_dir"
        cp "$final_dir/sarathi_log.log" "$LATEST_LOG"

        echo "$sweep_type,$model_safe,$num_requests,$qps,success,$inference_time,$wall_time,$final_dir" >> "$SUMMARY"

        echo "[SUCCESS] $final_dir"
    else
        local failed_dir="${final_dir}_failed_$(date +%Y%m%d_%H%M%S)"
        mv "$generated_dir" "$failed_dir"

        echo "$sweep_type,$model_safe,$num_requests,$qps,failed,$inference_time,$wall_time,$failed_dir" >> "$SUMMARY"

        rm -f "$marker"

        echo "[ERROR] Run failed. Output: $failed_dir"
        exit "$run_status"
    fi

    rm -f "$marker"
}

cd "$REPO" || exit 1

for model_pair in "${MODELS[@]}"; do
    model_name="${model_pair%%|*}"
    model_safe="${model_pair##*|}"

    # Setup 1: num_requests fixed = 10, qps varies 1..10.
    for qps in 1 2 3 4 5 6 7 8 9 10; do
        run_one "fixed_num_requests" "$model_name" "$model_safe" 10 "$qps"
    done

    # Setup 2: qps fixed = 1, num_requests varies 1..10.
    for num_requests in 1 2 3 4 5 6 7 8 9 10; do
        run_one "fixed_qps" "$model_name" "$model_safe" "$num_requests" 1
    done
done

echo ""
echo "Sweep complete."
echo "Summary: $SUMMARY"