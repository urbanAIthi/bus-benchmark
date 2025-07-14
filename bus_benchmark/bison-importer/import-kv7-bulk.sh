#!/bin/bash
FOLDER_IN=/mnt/nvme/benchmark/kv7
FOLDER_OUT=/mnt/nvme/benchmark/kv7_converted
THREADS=24
export TQDM_DISABLE=1
find "$FOLDER_IN" -name "*.csv.xz" | sort | parallel -j$THREADS --halt soon,fail=1 --progress --eta python ./import.py --mode kv7 --filters USERTIMINGPOINT TIMINGPOINT LINE --file {} --output-dir "$FOLDER_OUT"
