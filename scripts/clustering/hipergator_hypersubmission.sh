#!/bin/bash

run_id="CD_RGINT_v4_median"
spatial_aggregation="rgint"

N=100
cores=8
time="60:00:00"

for i in $(seq 1 "$N"); do

    repeat_id="$i"

    echo "Submitting repeat ${repeat_id} of run ID '${run_id}'"

    job_id=$(sbatch --parsable \
        --cpus-per-task "$cores" \
        --time="$time" \
        --job-name="${run_id}_repeat_${repeat_id}" \
        --mem-per-cpu=8gb \
        hipergator_submit_hyperoptimise-clusters_single.sh \
        "$run_id" \
        "$repeat_id" \
        "$cores" \
        "$spatial_aggregation")

    sleep 0.1

done
