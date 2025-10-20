#!/bin/bash

# ------------------------------------------------------
# Bash script to run Python pipeline with logging
# ------------------------------------------------------

# Exit immediately if any command fails
set -e
set -o pipefail  # safer error handling

# ------------------------------------------------------
# Define pipeline run variables
# ------------------------------------------------------

RUN_ID="compactness"                 
SPATIAL_AGGREGATION="rgint"

# ------------------------------------------------------
# Create a log directory
# ------------------------------------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./.log/${RUN_ID}_${SPATIAL_AGGREGATION}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ------------------------------------------------------
# Activate the conda environment
# ------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh  # change to where your anaconda3 lives
conda activate DENV-SEROTYPE-IMPUTATION

# ------------------------------------------------------
# Run Python scripts sequentially with logging
# ------------------------------------------------------

echo "[$(date)] Running find-clusters.py..." | tee -a "$LOG_DIR/pipeline.log"
python clustering/find-clusters.py \
    -ID "$RUN_ID" \
    -spatial_aggregation "$SPATIAL_AGGREGATION" \
    -n 250 \
    -threshold 50 \
    -compactness True \
    -biome False \
    -koppen False \
    -human_footprint False \
    -denv_100k_cumulative False \
    -denv_100k_DTW False \
    -indexP_DTW False \
    -serotypes_DTW False \
    > "$LOG_DIR/find-clusters.log" 2>&1

echo "[$(date)] Running fit-imputation-model.py..." | tee -a "$LOG_DIR/pipeline.log"
python bayesian-imputation-model/fit-imputation-model.py \
    -ID "$RUN_ID" \
    -spatial_aggregation "$SPATIAL_AGGREGATION" \
    -chains 4 \
    -p 1 \
    -q 1 \
    > "$LOG_DIR/fit-imputation-model.log" 2>&1

echo "[$(date)] Running visualise-fit.py..." | tee -a "$LOG_DIR/pipeline.log"
python bayesian-imputation-model/visualise-fit.py \
    -ID "$RUN_ID" \
    -spatial_aggregation "$SPATIAL_AGGREGATION" \
    -p 1 \
    -q 1 \
    > "$LOG_DIR/visualise-fit.log" 2>&1

# ------------------------------------------------------
# Deactivate environment
# ------------------------------------------------------

conda deactivate

echo "[$(date)] ✅ All scripts executed successfully!" | tee -a "$LOG_DIR/pipeline.log"
