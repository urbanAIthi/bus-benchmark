#!/bin/bash
set -euxo pipefail

source ../.env
read -ra LAU_IDS <<< "$LAU_IDS"

export KV6_TRANSFORMED KV6_FILTERED
parallel -j24 --halt now,fail=1 --eta 'echo "Processing {}"; python process.py --lau {} --input "$KV6_FILTERED/{}.csv.gz" --travel-output "$KV6_TRANSFORMED/travel_time/$(basename {}).csv.gz" --dwell-output "$KV6_TRANSFORMED/dwell_time/$(basename {}).csv.gz" --trajectory-output "$KV6_TRANSFORMED/trajectory/$(basename {}).csv.gz"' ::: "${LAU_IDS[@]}"
