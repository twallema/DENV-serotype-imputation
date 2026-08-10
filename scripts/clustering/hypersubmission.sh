#!/bin/bash

run_id="hyperoptimisation_rgint"
spatial_aggregation="rgint"

N=3
threads=2
time="01:00:00"

job_ids=()

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

    job_ids+=("$job_id")

done

dependency=$(IFS=:; echo "${job_ids[*]}")

echo "Submitted repeat jobs: ${dependency}"
echo "Submitting collection job after all repeats finish successfully"

sbatch \
    --dependency="afterok:${dependency}" \
    --job-name="${run_id}_results_collection" \
    collect_hyperoptimise_results.sh \
    "$run_id" \
    "$N"