#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS="${WORKERS:-24}"

echo "=== kv6_csv ==="
uv run python "$SCRIPT_DIR/import-csvs.py" --table kv6_csv --folder /mnt/nvme/benchmark/kv6_converted/ --workers "$WORKERS"

echo "=== kv7_line ==="
uv run python "$SCRIPT_DIR/import-csvs.py" --table kv7_line --folder /mnt/nvme/benchmark/kv7_line/ --workers "$WORKERS"

echo "=== kv7_localservicegrouppasstime ==="
uv run python "$SCRIPT_DIR/import-csvs.py" --table kv7_localservicegrouppasstime --folder /mnt/nvme/benchmark/kv7_localservicegrouppasstime/ --workers "$WORKERS"

echo "=== kv7_timingpoint ==="
uv run python "$SCRIPT_DIR/import-csvs.py" --table kv7_timingpoint --folder /mnt/nvme/benchmark/kv7_timingpoint/ --workers "$WORKERS"

echo "=== kv7_usertimingpoint ==="
uv run python "$SCRIPT_DIR/import-csvs.py" --table kv7_usertimingpoint --folder /mnt/nvme/benchmark/kv7_usertimingpoint/ --workers "$WORKERS"

echo ""
echo "All datasets imported."
