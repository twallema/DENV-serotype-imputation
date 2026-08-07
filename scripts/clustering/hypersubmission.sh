#!/bin/bash

N=2
threads=16
time="24:00:00"

for i in $(seq 1 $N); do

    ID="run_${i}"

    echo "Submitting ${ID}"

    sbatch \
        -c $threads \
        --time="${time}" \
        --job-name="hyper_${ID}" \
        submit_hyperoptimise-clusters_single.sh "${ID}" "${threads}"

done