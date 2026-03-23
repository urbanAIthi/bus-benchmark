#!/bin/bash
set -euo pipefail

source ../.env

export TQDM_DISABLE=1
find "$KV6_RAW" -name "*.csv.xz" | sort | parallel -j$THREADS --halt soon,fail=1 --progress --eta python ./convert.py --mode kv6 --file {} --output-dir "$KV6_CSV"
