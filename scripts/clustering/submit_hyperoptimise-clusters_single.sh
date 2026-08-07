#!/bin/bash
#SBATCH --account=arb24_0001
#SBATCH --partition=cac_cpu
#SBATCH --qos=longrun

# The ID and number of threads
ID=$1
threads=$2

if [ -z "$ID" ]; then
    echo "Error: no ID supplied"
    exit 1
fi

echo "Running hyperoptimisation with ID: ${ID}"

# Load Anaconda
module load anaconda3

# Activate conda environment
source /opt/ohpc/pub/software/anaconda3/etc/profile.d/conda.sh
conda activate DENV-SEROTYPE-IMPUTATION

unset PYTHONHOME
unset PYTHONPATH

# Run Python script
python hyperoptimise-clusters.py \
    -n_cores "${threads}" \
    -n_maxp 200 \
    -n_repeats 1 \
    -max_iterations_sa 10 \
    -spatial_aggregation rgint \
    -validation_bw 0.05 \
    -validation_n 2 \
    -ID "${ID}"

# Deactivate environment
conda deactivate