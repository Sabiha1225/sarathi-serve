#!/usr/bin/env bash
set -uo pipefail

# chmod +x osdi-experiments/graphs/run_all_arxiv_metric_analysis.sh
# ./osdi-experiments/graphs/run_all_arxiv_metric_analysis.sh

REPO="/home/sabiha/sarathi-serve"
ANALYZER="$REPO/osdi-experiments/graphs/analyze_sarathi_metric.py"
TIMES_CSV="$REPO/osdi-experiments/graphs/time_arxiv_sarathi.csv"
OUTPUT_ROOT="$REPO/benchmark_output/arxiv"
SUMMARY="$REPO/osdi-experiments/graphs/arxiv_metric_analysis_summary.csv"

echo "model,chunk_size,qps,total_time,status,sequence_metrics_csv,output_dir" > "$SUMMARY"

{
    read -r header

    while IFS=',' read -r dataset model chunk_size inference requests request_arrival || [[ -n "$dataset" ]]; do
        [[ -z "$model" ]] && continue

        total_time="${inference% s}"
        qps=$(echo "$request_arrival" | awk '{print $2}')

        RUN_DIR="$OUTPUT_ROOT/arxiv_chunk_${chunk_size}_poison_${qps}_qps_${model}"
        SEQ_CSV="$RUN_DIR/replica_0/sequence_metrics.csv"

        echo ""
        echo "============================================================"
        echo "Analyzing model=$model chunk_size=$chunk_size qps=$qps total_time=$total_time"
        echo "CSV: $SEQ_CSV"
        echo "============================================================"

        if [[ ! -f "$SEQ_CSV" ]]; then
            echo "[MISSING] $SEQ_CSV"
            echo "$model,$chunk_size,$qps,$total_time,missing,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
            continue
        fi

        python "$ANALYZER" "$SEQ_CSV" \
            --total-time "$total_time" \
            --visualize \
            --json

        status=$?

        if [[ "$status" -eq 0 ]]; then
            echo "$model,$chunk_size,$qps,$total_time,success,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
        else
            echo "$model,$chunk_size,$qps,$total_time,failed,$SEQ_CSV,$RUN_DIR" >> "$SUMMARY"
        fi
    done
} < "$TIMES_CSV"

echo ""
echo "Done."
echo "Summary: $SUMMARY"
