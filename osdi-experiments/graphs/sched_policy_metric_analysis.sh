#!/usr/bin/env bash
set -uo pipefail

# chmod +x osdi-experiments/graphs/sched_policy_metric_analysis.sh
# ./osdi-experiments/graphs/sched_policy_metric_analysis.sh

REPO="/home/sabiha/sarathi_observation2"
ANALYZER="$REPO/osdi-experiments/graphs/analyze_sarathi_metric.py"
TIMES_CSV="$REPO/osdi-experiments/graphs/sched_policy_time.csv"
OUTPUT_ROOT="$REPO/benchmark_output/figure-1"
SUMMARY="$REPO/osdi-experiments/graphs/sched_policy_metric_analysis_summary.csv"

CHUNK_SIZE=512
QPS=3
NUM_REQUESTS=300

# model_safe|model_name  (model_safe matches the sweep script's directory naming,
# model_name matches the sched_policy_time.csv column)
MODELS=(
    "llama2_7b|meta-llama/Llama-2-7b-hf"
    "yi_6b|01-ai/Yi-6B"
    "mixtral_7b_8expert|DiscoResearch/mixtral-7b-8expert"
)

DATASETS=(
    "long_long"
    "long_short"
    "mixed"
    "short_long"
    "short_short"
)

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

echo "model_safe,dataset,policy,total_time,status,sequence_metrics_csv,output_dir" > "$SUMMARY"

lookup_total_time() {
    local model_name="$1"
    local dataset="$2"
    local policy="$3"

    awk -F',' -v m="$model_name" -v d="$dataset" -v p="$policy" '
        NR==1 { next }
        $1==m && $2==d && $3==p { print $4; found=1; exit }
        END { if (!found) exit 1 }
    ' "$TIMES_CSV"
}

for model_pair in "${MODELS[@]}"; do
    model_safe="${model_pair%%|*}"
    model_name="${model_pair##*|}"

    for dataset in "${DATASETS[@]}"; do
        for policy in "${POLICIES[@]}"; do

            RUN_DIR="$OUTPUT_ROOT/${model_safe}_${dataset}_sched_policy_${policy}_chunk${CHUNK_SIZE}_qps${QPS}_n${NUM_REQUESTS}"
            SEQ_CSV="$RUN_DIR/replica_0/sequence_metrics.csv"

            echo ""
            echo "============================================================"
            echo "Analyzing model=$model_safe dataset=$dataset policy=$policy"
            echo "CSV: $SEQ_CSV"
            echo "============================================================"

            if [[ ! -f "$SEQ_CSV" ]]; then
                echo "[MISSING] $SEQ_CSV"
                echo "$model_safe,$dataset,$policy,,missing_csv,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
                continue
            fi

            total_time=$(lookup_total_time "$model_name" "$dataset" "$policy")
            if [[ -z "$total_time" ]]; then
                echo "[WARN] No total_time found in $TIMES_CSV for $model_name/$dataset/$policy — letting analyzer estimate it from data"
                python "$ANALYZER" "$SEQ_CSV" --visualize --json
            else
                echo "Using total_time=$total_time from $TIMES_CSV"
                python "$ANALYZER" "$SEQ_CSV" --total-time "$total_time" --visualize --json
            fi

            status=$?

            if [[ "$status" -eq 0 ]]; then
                echo "$model_safe,$dataset,$policy,$total_time,success,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
            else
                echo "$model_safe,$dataset,$policy,$total_time,failed,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
            fi

        done
    done
done

echo ""
echo "Done."
echo "Summary: $SUMMARY"
