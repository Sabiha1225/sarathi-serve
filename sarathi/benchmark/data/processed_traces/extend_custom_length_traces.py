#!/usr/bin/env python3

import random
from pathlib import Path
import pandas as pd

# NOTE: fixed to point at sarathi_observation2, unlike the original
# custom_data_creation.py which still points at sarathi-serve.
# sarathi/benchmark/data/processed_traces/extend_custom_length_traces.py

OUT_DIR = Path("/home/sabiha/sarathi_observation2/sarathi/benchmark/data/processed_traces/custom_length_traces")

TARGET_TOTAL = 300
NEW_ROWS_SEED = 142  # deliberately different from the original script's seed=42,
                      # so the 270 new draws don't just replay the same sequence


def make_rows(num_requests, prefill_min, prefill_max, decode_min, decode_max):
    rows = []
    for _ in range(num_requests):
        prefill = random.randint(prefill_min, prefill_max)
        decode = random.randint(decode_min, decode_max)
        total = prefill + decode
        pd_ratio = prefill / decode if decode > 0 else 0.0
        rows.append({
            "num_prefill_tokens": prefill,
            "num_decode_tokens": decode,
            "num_total_tokens": total,
            "pd_ratio": pd_ratio,
        })
    return rows


def extend_file(filename, prefill_min, prefill_max, decode_min, decode_max):
    path = OUT_DIR / filename
    existing_df = pd.read_csv(path)

    if len(existing_df) >= TARGET_TOTAL:
        print(f"skip {filename}: already has {len(existing_df)} rows (>= {TARGET_TOTAL})")
        return

    num_new = TARGET_TOTAL - len(existing_df)
    new_rows = make_rows(num_new, prefill_min, prefill_max, decode_min, decode_max)
    new_df = pd.DataFrame(new_rows)

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(path, index=False)
    print(f"extended {filename}: {len(existing_df)} -> {len(combined_df)} rows")


def extend_mixed_file(filename):
    path = OUT_DIR / filename
    existing_df = pd.read_csv(path)

    if len(existing_df) >= TARGET_TOTAL:
        print(f"skip {filename}: already has {len(existing_df)} rows (>= {TARGET_TOTAL})")
        return

    num_new = TARGET_TOTAL - len(existing_df)
    # preserve the original 8:8:7:7 category ratio from custom_data_creation.py
    n1 = round(num_new * 8 / 30)
    n2 = round(num_new * 8 / 30)
    n3 = round(num_new * 7 / 30)
    n4 = num_new - n1 - n2 - n3  # remainder soaks up rounding

    new_rows = []
    new_rows.extend(make_rows(n1, 1, 512, 1, 300))
    new_rows.extend(make_rows(n2, 1, 512, 300, 8000))
    new_rows.extend(make_rows(n3, 512, 10000, 1, 512))
    new_rows.extend(make_rows(n4, 512, 10000, 512, 8000))
    random.shuffle(new_rows)

    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(path, index=False)
    print(f"extended {filename}: {len(existing_df)} -> {len(combined_df)} rows")


if __name__ == "__main__":
    random.seed(NEW_ROWS_SEED)

    extend_file("short_input_512_short_output_300.csv", 1, 512, 1, 300)
    extend_file("short_input_512_long_output_8000.csv", 1, 512, 300, 8000)
    extend_file("long_input_10000_short_output_512.csv", 512, 10000, 1, 512)
    extend_file("long_input_10000_long_output_8000.csv", 512, 10000, 512, 8000)
    extend_mixed_file("mixed_short_long_30_requests.csv")
