#!/bin/bash
#SBATCH --account=epi
#SBATCH --qos=epi-b

# The IDs and number of cores
run_id=$1
repeat_id=$2
cores=$3
spatial_aggregation=$4

echo "Running repeat ${repeat_id} of clustering hyperoptimisation with ID '${run_id}'"

# Load Anaconda
module purge
module load conda

# Activate conda environment
conda activate DENV-SEROTYPE-IMPUTATION

# Run Python script
python hyperoptimise-clusters.py \
    --n_cores "${cores}" \
    --n_maxp 250 \
    --max_iterations_sa 10 \
    --spatial_aggregation "${spatial_aggregation}" \
    --validation_n 279 \
    --run_id "${run_id}" \
    --repeat_id "${repeat_id}"

# Deactivate environment
conda deactivate