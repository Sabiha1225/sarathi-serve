#!/usr/bin/env python3
"""
Parse log/vllm_baseline.txt into a CSV with columns: model, qps, total_time_sec.

python log_processing/parse_vllm_baseline_timing_logs.py log/vllm_baseline.txt log_csv/vllm_baseline.csv
"""
import csv
import re
import sys

MODEL_RE = re.compile(r"^#\s*model=(\S+)\s+vLLM")
QPS_RE = re.compile(r"^#\s*vLLM Poisson\s+(\d+)\s+request per seconds")
RESULT_RE = re.compile(r"Total time taken:\s*([\d.]+)\s*seconds")


def parse_log(log_path):
    rows = []
    model = None
    qps = None

    with open(log_path) as f:
        for line in f:
            m = MODEL_RE.match(line)
            if m:
                model = m.group(1)
                continue

            m = QPS_RE.match(line)
            if m:
                qps = int(m.group(1))
                continue

            m = RESULT_RE.search(line)
            if m:
                if model is None or qps is None:
                    print(f"[warn] result line with no preceding model/qps: {line.strip()}")
                    continue
                rows.append({
                    "model": model,
                    "qps": qps,
                    "total_time_sec": float(m.group(1)),
                })

    return rows


def main():
    if len(sys.argv) != 3:
        print("usage: parse_vllm_baseline_log.py <log_path> <output_csv_path>")
        sys.exit(1)

    log_path, output_csv_path = sys.argv[1], sys.argv[2]
    rows = parse_log(log_path)

    if not rows:
        print(f"[warn] no rows parsed from {log_path}")
        return

    fieldnames = ["model", "qps", "total_time_sec"]
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output_csv_path}")


if __name__ == "__main__":
    main()
