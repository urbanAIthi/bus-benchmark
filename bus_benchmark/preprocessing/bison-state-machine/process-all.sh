#!/bin/bash
set -euxo pipefail

source ../.env

export KV6_TRANSFORMED KV6_FILTERED
parallel -j24 --halt now,fail=1 --eta 'echo "Processing {}"; python process.py --lau {} --input "$KV6_TRANSFORMED/{}.csv.gz" --travel-output "$KV6_TRANSFORMED/travel_time/$(basename {}).csv.gz" --dwell-output "$KV6_TRANSFORMED/dwell_time/$(basename {}).csv.gz"' ::: "${LAUS[@]}"
