#!/usr/bin/env python3
"""
Parse the sarathi_baseline_fixed_chunk.txt / sarathi_baseline_dynamic_chunk.txt
sweep logs into a CSV with columns: model, qps, chunk_size, dynamic_chunking, total_time_sec.

python log_processing/parse_sarathi_baseline_timing_logs.py log/sarathi_baseline_fixed_chunk.txt log_csv/sarathi_baseline_fixed_chunk.csv
python log_processing/parse_sarathi_baseline_timing_logs.py log/sarathi_baseline_dynamic_chunk.txt log_csv/sarathi_baseline_dynamic_chunk.csv
"""
import csv
import re
import sys

HEADER_RE = re.compile(
    r"^#\s*model=(\S+)\s+Poisson\s+(\d+)\s+request per seconds\s+"
    r"\(chunk_size=(\d+),\s*dynamic=(true|false)\)"
)
RESULT_RE = re.compile(r"Total time taken:\s*([\d.]+)\s*seconds")


def parse_log(log_path):
    rows = []
    current = None

    with open(log_path) as f:
        for line in f:
            header_match = HEADER_RE.search(line)
            if header_match:
                model, qps, chunk_size, dynamic = header_match.groups()
                current = {
                    "model": model,
                    "qps": int(qps),
                    "chunk_size": int(chunk_size),
                    "dynamic_chunking": dynamic == "true",
                }
                continue

            result_match = RESULT_RE.search(line)
            if result_match:
                if current is None:
                    print(f"[warn] result line with no preceding header: {line.strip()}")
                    continue
                row = dict(current)
                row["total_time_sec"] = float(result_match.group(1))
                rows.append(row)
                current = None

    return rows


def main():
    if len(sys.argv) != 3:
        print("usage: parse_sarathi_baseline_timing_logs.py <log_path> <output_csv_path>")
        sys.exit(1)

    log_path, output_csv_path = sys.argv[1], sys.argv[2]
    rows = parse_log(log_path)

    if not rows:
        print(f"[warn] no rows parsed from {log_path}")
        return

    fieldnames = ["model", "qps", "chunk_size", "dynamic_chunking", "total_time_sec"]
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output_csv_path}")


if __name__ == "__main__":
    main()
