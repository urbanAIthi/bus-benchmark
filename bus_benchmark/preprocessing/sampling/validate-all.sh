#!/bin/bash
set -euxo pipefail

source ../.env

export TQDM_DISABLE=1
export TT_INPUT_FOLDER DT_INPUT_FOLDER TT_OUTPUT_FOLDER DT_OUTPUT_FOLDER

# use memsuspend to swap processes to disk when necessary
# this is needed because some of jobs use insane amounts of memory
parallel -j24 --memsuspend 8G --halt now,fail=1 --eta python validate.py --tt-input "$KV6_TRANSFORMED/travel_time/{}.csv.gz" --dt-input "$KV6_TRANSFORMED/dwell_time/{}.csv.gz" --tt-output "$KV6_VALIDATED/travel_time/{}.parquet" --dt-output "$KV6_VALIDATED/dwell_time/{}.parquet" ::: "${LAU_IDS[@]}"
