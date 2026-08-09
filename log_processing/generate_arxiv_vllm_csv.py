#!/usr/bin/env python3

import re
import csv

SRC = "/home/sabiha/sarathi-serve/log/time_arxiv_vllm.txt"
OUT = "/home/sabiha/sarathi-serve/log/time_arxiv_vllm.csv"

# python3 log_processing/generate_arxiv_vllm_csv.py

lines = open(SRC).read().splitlines()

rows = []
model = None

for line in lines:
    m = re.match(r'# model=(\S+) vLLM', line)
    if m:
        model = m.group(1)
        continue

    m = re.match(r'# vLLM Poisson (\d+) request per seconds', line)
    if m:
        qps = m.group(1)
        continue

    m = re.search(r'processing (\d+) \(.*Total time taken: ([\d.]+) seconds', line)
    if m and model:
        requests = m.group(1)
        inference_time = m.group(2)
        rows.append({
            'Dataset': 'Arxiv',
            'Model': model,
            'Inference': f'{inference_time} s',
            'Requests': requests,
            'Request Arrival': f'Poisson {qps} QPS',
        })

fieldnames = ['Dataset', 'Model', 'Inference', 'Requests', 'Request Arrival']
with open(OUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'wrote {len(rows)} rows to {OUT}')
