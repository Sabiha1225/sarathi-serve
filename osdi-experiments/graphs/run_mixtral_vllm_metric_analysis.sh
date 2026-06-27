#!/usr/bin/env bash
set -uo pipefail

# chmod +x osdi-experiments/graphs/run_mixtral_vllm_metric_analysis.sh
# ./osdi-experiments/graphs/run_mixtral_vllm_metric_analysis.sh

REPO="/home/sabiha/sarathi-serve"
ANALYZER="$REPO/osdi-experiments/graphs/analyze_sarathi_metric.py"
TIMES_CSV="$REPO/osdi-experiments/graphs/mixtral_vllm_times.csv"
OUTPUT_ROOT="$REPO/benchmark_output/figure-1"
SUMMARY="$REPO/osdi-experiments/graphs/mixtral_vllm_metric_analysis_summary.csv"

echo "qps,total_time,status,sequence_metrics_csv,output_dir" > "$SUMMARY"

{
    read -r header

    while IFS=',' read -r qps total_time || [[ -n "$qps" ]]; do
        RUN_DIR="$OUTPUT_ROOT/arxiv_vllm_poison_${qps}_qps_mixtral"
        SEQ_CSV="$RUN_DIR/replica_0/sequence_metrics.csv"

        echo ""
        echo "============================================================"
        echo "Analyzing vLLM qps=$qps total_time=$total_time"
        echo "CSV: $SEQ_CSV"
        echo "============================================================"

        if [[ ! -f "$SEQ_CSV" ]]; then
        
            echo "[MISSING] $SEQ_CSV"
            echo "$qps,$total_time,missing,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
            continue
        fi

        python "$ANALYZER" "$SEQ_CSV" \
            --total-time "$total_time" \
            --visualize \
            --json

        status=$?

        if [[ "$status" -eq 0 ]]; then
            echo "$qps,$total_time,success,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
        else
            echo "$qps,$total_time,failed,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
        fi
    done
} < "$TIMES_CSV"

echo ""
echo "Done."
echo "Summary: $SUMMARY"