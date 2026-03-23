#!/bin/bash
set -euo pipefail

source ../.env

export TQDM_DISABLE=1
find "$KV7_RAW" -name "*.csv.xz" | sort | parallel -j$THREADS --halt soon,fail=1 --progress --eta python ./convert.py --mode kv7 --filters USERTIMINGPOINT TIMINGPOINT LINE --file {} --output-dir "$KV7_CSV"
