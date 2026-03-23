#!/bin/bash
set -euo pipefail

source ../.env

echo "Importing kv6_csv"
uv run python "./import.py" --table kv6_csv --folder "$KV6_CSV" --workers "$THREADS"

echo "Importing kv7_line"
uv run python "./import.py" --table kv7_line --folder "$KV7_CSV/LINE/" --workers "$THREADS"

echo "Importing kv7_localservicegrouppasstime"
uv run python "./import.py" --table kv7_localservicegrouppasstime --folder "$KV7_CSV/LOCALSERVICEGROUPPASSTIME/" --workers "$THREADS"

echo "Importing kv7_timingpoint"
uv run python "./import.py" --table kv7_timingpoint --folder "$KV7_CSV/TIMINGPOINT/" --workers "$THREADS"

echo "Importing kv7_usertimingpoint"
uv run python "./import.py" --table kv7_usertimingpoint --folder "$KV7_CSV/USERTIMINGPOINT/" --workers "$THREADS"
