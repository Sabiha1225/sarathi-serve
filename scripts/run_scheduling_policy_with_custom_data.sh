#!/usr/bin/env bash
set -uo pipefail

# chmod +x scripts/run_scheduling_policy_with_custom_data.sh
# nohup ./scripts/run_scheduling_policy_with_custom_data.sh > log/sched_policy_sweep_all_configs.log 2>&1 &
# echo $! > log/sched_policy_sweep_all_configs.pid
# tail -f log/sched_policy_sweep_all_configs.log

REPO="/home/sabiha/sarathi_observation2"
MAIN="$REPO/sarathi/benchmark/main.py"
CONSTANTS_FILE="$REPO/sarathi/benchmark/constants.py"
OUTPUT_ROOT="$REPO/benchmark_output/figure-1"
TIME_FILE="$REPO/log/sched_policy_time.txt"
LATEST_LOG="$REPO/log/sarathi_log.log"
SUMMARY="$OUTPUT_ROOT/sched_policy_sweep_all_configs_summary.csv"

CHUNK_SIZE=512
QPS=3
NUM_REQUESTS=300

# name|safe_label
MODELS=(
    "meta-llama/Llama-2-7b-hf|llama2_7b"
    "01-ai/Yi-6B|yi_6b"
    "DiscoResearch/mixtral-7b-8expert|mixtral_7b_8expert"
)

# Tier 1 (core) + Tier 2 (secondary) policies, per the scoping discussed earlier.
POLICIES=(
    "fcfs"
    "sjf"
    "ljf"
    "prompt_len_sjf"
    "prefill_aging_fairness"
    "vtc_fairness"
    "skip_join_mlfq"
    "least_laxity_first"
    "hybrid"
    "adaptive_io_aging"
)

# label|yml_filename
DATASETS=(
    "long_long|custom_data_long_input_long_output_llama_sarathi.yml"
    "long_short|custom_data_long_input_short_output_llama_sarathi.yml"
    "mixed|custom_data_mixed_long_short_llama_sarathi.yml"
    "short_long|custom_data_short_input_long_output_llama_sarathi.yml"
    "short_short|custom_data_short_input_short_output_llama_sarathi.yml"
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
    echo "model,dataset,policy,chunk_size,qps,num_requests,status,inference_time_sec,wall_time_sec,output_dir" > "$SUMMARY"
fi

# Point constants.py's DEFAULT_CONFIG_FILE at the given yml, commenting out
# whatever line is currently active. This is the only way to select a config
# file, since main.py takes no --config_file flag.
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

run_one() {
    local model_name="$1"
    local model_safe="$2"
    local dataset_label="$3"
    local policy="$4"

    local run_name="${model_safe}_${dataset_label}_sched_policy_${policy}_chunk${CHUNK_SIZE}_qps${QPS}_n${NUM_REQUESTS}"
    local final_dir="$OUTPUT_ROOT/$run_name"

    if [[ -d "$final_dir" ]]; then
        echo "[SKIP] Already exists: $final_dir"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "Running model=$model_name dataset=$dataset_label policy=$policy chunk=$CHUNK_SIZE qps=$QPS n=$NUM_REQUESTS"
    echo "Output: $run_name"
    echo "============================================================"

    echo "" >> "$TIME_FILE"
    echo "Policy sweep: model=$model_name dataset=$dataset_label policy=$policy chunk=$CHUNK_SIZE qps=$QPS n=$NUM_REQUESTS" >> "$TIME_FILE"

    local marker gpu_tmp top_tmp log_tmp
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

    top -d 4.0 -u "$(whoami)" -b > "$top_tmp" 2>&1 &
    TOP_PID=$!

    set +e
    python "$MAIN" \
        --model_name "$model_name" \
        --sarathi_scheduler_chunk_size "$CHUNK_SIZE" \
        --poisson_request_interval_generator_qps "$QPS" \
        --synthetic_request_generator_num_requests "$NUM_REQUESTS" \
        --replica_scheduler_policy_name "$policy" \
        2>&1 | tee "$log_tmp"
    local run_status="${PIPESTATUS[0]}"
    set -e

    local end_time wall_time
    end_time="$(date +%s)"
    wall_time="$((end_time - start_time))"

    cleanup_monitors
    sleep 25

    local generated_dir
    generated_dir=$(
        find "$OUTPUT_ROOT" \
            -mindepth 1 -maxdepth 1 -type d \
            -name '20??-??-??_*' \
            -newer "$marker" \
            -printf '%T@ %p\n' |
            sort -nr | head -n 1 | cut -d' ' -f2-
    )

    local inference_time
    inference_time=$(
        grep -oE 'Total time taken: [0-9.]+ seconds' "$log_tmp" |
            tail -n 1 | awk '{print $4}'
    )
    [[ -z "$inference_time" ]] && inference_time="NA"

    if [[ -z "$generated_dir" || ! -d "$generated_dir" ]]; then
        local failed_dir="${final_dir}_failed_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$failed_dir"
        mv "$gpu_tmp" "$failed_dir/nvidia-smi-output.csv"
        mv "$top_tmp" "$failed_dir/top_output.txt"
        mv "$log_tmp" "$failed_dir/sarathi_log.log"
        echo "$model_safe,$dataset_label,$policy,$CHUNK_SIZE,$QPS,$NUM_REQUESTS,failed,$inference_time,$wall_time,$failed_dir" >> "$SUMMARY"
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
        echo "$model_safe,$dataset_label,$policy,$CHUNK_SIZE,$QPS,$NUM_REQUESTS,success,$inference_time,$wall_time,$final_dir" >> "$SUMMARY"
        echo "[SUCCESS] $final_dir"
    else
        local failed_dir="${final_dir}_failed_$(date +%Y%m%d_%H%M%S)"
        mv "$generated_dir" "$failed_dir"
        echo "$model_safe,$dataset_label,$policy,$CHUNK_SIZE,$QPS,$NUM_REQUESTS,failed,$inference_time,$wall_time,$failed_dir" >> "$SUMMARY"
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

    for dataset_pair in "${DATASETS[@]}"; do
        dataset_label="${dataset_pair%%|*}"
        yml_name="${dataset_pair##*|}"

        set_active_config "$yml_name"

        for policy in "${POLICIES[@]}"; do
            run_one "$model_name" "$model_safe" "$dataset_label" "$policy"
        done
    done
done

echo ""
echo "Sweep complete."
echo "Summary: $SUMMARY"
