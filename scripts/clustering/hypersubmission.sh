#!/bin/bash

run_id="hyperoptimisation_rgint"
spatial_aggregation="rgint"

N=30
threads=16
time="72:00:00"

for i in $(seq 1 "$N"); do

    repeat_id="$i"

    echo "Submitting repeat ${repeat_id} of run ID '${run_id}'"

    job_id=$(sbatch --parsable \
        -c "$threads" \
        --time="$time" \
        --job-name="${run_id}_repeat_${repeat_id}" \
        submit_hyperoptimise-clusters_single.sh \
        "$run_id" \
        "$repeat_id" \
        "$threads" \
        "$spatial_aggregation")

done
