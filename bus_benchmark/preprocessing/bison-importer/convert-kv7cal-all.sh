#!/bin/bash
set -euo pipefail

source ../.env

# The archive keeps a .hash.csv.xz sidecar next to every feed file; those are checksums,
# not messages, so exclude them.
export TQDM_DISABLE=1
find "$KV7CAL_RAW" -name "KV7calendar_*.csv.xz" ! -name "*.hash.csv.xz" | sort \
    | parallel -j$THREADS --halt soon,fail=1 --progress --eta \
        uv run python ./convert.py --mode kv7calendar \
            --filters LOCALSERVICEGROUPVALIDITY --file {} --output-dir "$KV7CAL_CSV"
