#!/bin/bash
set -euxo pipefail

RAW_INPUT_FOLDER=/mnt/nvme/sql/kv6_filtered_v2
REFERENCE_INPUT_FOLDER=/mnt/nvme/sql/kv6_validated_v2/travel_time_filtered
OUTPUT_FOLDER=/mnt/nvme/sql/kv6_extracted_trajectories_v2
LAU_IDS=(GM0599 GM0518 GM0344 GM0546 GM0503 GM1930 GM0590 GM1842 GM0281 GM0059 GM0047 GM0629 GM0312 GM1969 GM1731 GM1950 GM1681 GM1690)

export TQDM_DISABLE=1
export RAW_INPUT_FOLDER REFERENCE_INPUT_FOLDER OUTPUT_FOLDER

parallel -j24 --halt now,fail=1 --eta 'echo "Processing {}"; python extract-trajectories.py --reference "$REFERENCE_INPUT_FOLDER/{}.csv" --raw "$RAW_INPUT_FOLDER/{}.csv.gz" --output "$OUTPUT_FOLDER/$(basename {}).csv"' ::: "${LAU_IDS[@]}"
