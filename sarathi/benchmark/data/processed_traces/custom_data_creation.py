#!/usr/bin/env python3

import random
from pathlib import Path
import pandas as pd

random.seed(42)

OUT_DIR = Path("/home/sabiha/sarathi-serve/sarathi/benchmark/data/processed_traces/custom_length_traces")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_REQUESTS = 30


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


def write_csv(filename, rows):
    path = OUT_DIR / filename
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"wrote {path}")


# 1. short input, short output
# max input 512, max output 300
write_csv(
    "short_input_512_short_output_300.csv",
    make_rows(
        NUM_REQUESTS,
        prefill_min=1,
        prefill_max=512,
        decode_min=1,
        decode_max=300,
    ),
)

# 2. short input, long output
# max input 512, max output 8000
write_csv(
    "short_input_512_long_output_8000.csv",
    make_rows(
        NUM_REQUESTS,
        prefill_min=1,
        prefill_max=512,
        decode_min=300,
        decode_max=8000,
    ),
)

# 3. long input, short output
# max input 10000, max output 512
write_csv(
    "long_input_10000_short_output_512.csv",
    make_rows(
        NUM_REQUESTS,
        prefill_min=512,
        prefill_max=10000,
        decode_min=1,
        decode_max=512,
    ),
)

# 4. long input, long output
# max input 10000, max output 8000
write_csv(
    "long_input_10000_long_output_8000.csv",
    make_rows(
        NUM_REQUESTS,
        prefill_min=512,
        prefill_max=10000,
        decode_min=512,
        decode_max=8000,
    ),
)

# 5. mixed workload: equal mix from all four categories
mixed_rows = []
mixed_rows.extend(make_rows(8, 1, 512, 1, 300))
mixed_rows.extend(make_rows(8, 1, 512, 300, 8000))
mixed_rows.extend(make_rows(7, 512, 10000, 1, 512))
mixed_rows.extend(make_rows(7, 512, 10000, 512, 8000))
random.shuffle(mixed_rows)

write_csv(
    "mixed_short_long_30_requests.csv",
    mixed_rows,
)