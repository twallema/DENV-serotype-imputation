#!/bin/bash
#SBATCH --account=arb24_0001
#SBATCH --partition=cac_cpu
#SBATCH -c 1
#SBATCH --time=00:10:00

run_id="$1"
N=$2

mkdir -p "../../data/interim/clustering_pipeline/${run_id}/results"

for ((i=1; i<=N; i++)); do
    cp \
        "../../data/interim/clustering_pipeline/${run_id}/repeat_${i}/results.csv" \
        "../../data/interim/clustering_pipeline/${run_id}/results/repeat_${i}.csv"
done