#!/bin/bash

# Figure output is going in pipeline_output -- figure out why and how to change since this is not the pipeline output but rather find-clusters testing

# Variation of information - will tell you hwo different clusters are

# Exit immediately if any command fails
set -e
set -o pipefail  # safer error handling

SPATIAL_AGGREGATION="rgint"

SA_VALUES=(10 20 50)
N_VALUES=(100 500 1000)

COVARIATE_SETS=("none" "two_covariates")

RUN_ID=1

# Log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./.log/${RUN_ID}_${SPATIAL_AGGREGATION}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

source /opt/anaconda3/etc/profile.d/conda.sh  # change to where your anaconda3 lives
conda activate DENV-SEROTYPE-IMPUTATION

for SA in "${SA_VALUES[@]}"; do
    for N in "${N_VALUES[@]}"; do 
        for COV in "${COVARIATE_SETS[@]}"; do

            echo "========================================"
            echo "[$(date)] Running combination ID: $RUN_ID" | tee -a "$LOG_DIR/find_clusters_tests.log"
            echo "Simulated Annealing Steps: $SA"
            echo "Number of Clustering Runs: $N"
            echo "Covariates: $COV"
            echo "========================================"

            COMPACTNESS=False
            BIOME=False
            KOPPEN=False
            HUMAN_FOOTPRINT=False
            DENV_100K_CUMULATIVE=False
            DENV_100K_DTW=False
            INDEXP_DTW=False
            SEROTYPES_DTW=False

            if [ "$COV"=="two_covariates" ]; then
                DENV_100K_DTW=TRUE
                INDEPXP_DTW=TRUE
            fi

            LOG_FILE="$LOG_DIR/run_${RUN_ID}_SA${SA}_N${N}_${COV}.log"

            python find-clusters.py \
                -ID "$RUN_ID" \
                -spatial_aggregation "$SPATIAL_AGGREGATION" \
                -n "$N" \
                -max_iterations_sa "$SA" \
                -threshold 35 \
                -compactness $COMPACTNESS \
                -biome $BIOME \
                -koppen $KOPPEN \
                -human_footprint $HUMAN_FOOTPRINT \
                -denv_100k_cumulative $DENV_100K_CUMULATIVE \
                -denv_100k_DTW $DENV_100K_DTW \
                -indexP_DTW $INDEXP_DTW \
                > "$LOG_FILE" 2>&1

            ((RUN_ID++))

        done
    done
done

conda deactivate

echo "[$(date)] ✅ All scripts executed successfully!" | tee -a "$LOG_DIR/find_clusters_tests.log"
