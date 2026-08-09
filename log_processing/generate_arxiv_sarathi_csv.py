#!/usr/bin/env python3

import re
import csv

# python3 log_processing/generate_arxiv_sarathi_csv.py

SRC = "/home/sabiha/sarathi-serve/log/time_arxiv_sarathi.txt"
OUT = "/home/sabiha/sarathi-serve/log/time_arxiv_sarathi.csv"

lines = open(SRC).read().splitlines()

rows = []
model = None
qps = None
chunk_size = None

for line in lines:
    m = re.match(r'# model=(\S+) Poisson (\d+) request per seconds', line)
    if m:
        model = m.group(1)
        qps = m.group(2)
        continue

    m = re.match(r'# chunk_size=(\d+)', line)
    if m:
        chunk_size = m.group(1)
        continue

    m = re.search(r'processing (\d+) \(.*Total time taken: ([\d.]+) seconds', line)
    if m and model and chunk_size:
        requests = m.group(1)
        inference_time = m.group(2)
        rows.append({
            'Dataset': 'Arxiv',
            'Model': model,
            'Chunk Size': chunk_size,
            'Inference': f'{inference_time} s',
            'Requests': requests,
            'Request Arrival': f'Poisson {qps} QPS',
        })
        chunk_size = None  # require a fresh chunk_size line before the next match

fieldnames = ['Dataset', 'Model', 'Chunk Size', 'Inference', 'Requests', 'Request Arrival']
with open(OUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'wrote {len(rows)} rows to {OUT}')
