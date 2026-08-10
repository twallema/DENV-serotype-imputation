#!/bin/bash
#SBATCH --account=arb24_0001
#SBATCH --partition=cac_cpu
#SBATCH --qos=longrun

# The IDs and number of threads
run_id=$1
repeat_id=$2
threads=$3
spatial_aggregation=$4

echo "Running repeat ${repeat_id} of clustering hyperoptimisation with ID '${run_id}'"

# Load Anaconda
module load anaconda3

# Activate conda environment
source /opt/ohpc/pub/software/anaconda3/etc/profile.d/conda.sh
conda activate DENV-SEROTYPE-IMPUTATION

unset PYTHONHOME
unset PYTHONPATH

# Run Python script
python hyperoptimise-clusters.py \
    --n_cores "${threads}" \
    --n_maxp 250 \
    --max_iterations_sa 10 \
    --spatial_aggregation "${spatial_aggregation}" \
    --validation_bw 0.05 \
    --validation_n 2 \
    --run_id "${run_id}" \
    --repeat_id "${repeat_id}"

# Deactivate environment
conda deactivate