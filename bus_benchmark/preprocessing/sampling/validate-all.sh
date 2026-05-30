#!/bin/bash
set -euxo pipefail

source ../.env
read -ra LAU_IDS <<< "$LAU_IDS"

export TQDM_DISABLE=1

mkdir -p "$KV6_VALIDATED/travel_time"
mkdir -p "$KV6_VALIDATED/dwell_time"
mkdir -p "$KV6_VALIDATED/trajectory"

# use memsuspend to swap processes to disk when necessary
# this is needed because some of jobs use insane amounts of memory
parallel -j24 --memsuspend 16G --halt now,fail=1 --eta python validate.py --tt-input "$KV6_TRANSFORMED/travel_time/{}.csv.gz" --dt-input "$KV6_TRANSFORMED/dwell_time/{}.csv.gz" --traj-input "$KV6_TRANSFORMED/trajectory/{}.csv.gz" --tt-output "$KV6_VALIDATED/travel_time/{}.parquet" --dt-output "$KV6_VALIDATED/dwell_time/{}.parquet" --traj-output "$KV6_VALIDATED/trajectory/{}.parquet" ::: "${LAU_IDS[@]}"
