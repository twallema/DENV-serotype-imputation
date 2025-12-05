#!/bin/bash

# Exit immediately if any command fails
set -e
set -o pipefail  # safer error handling

SPATIAL_AGGREGATION="rgint"

SA_VALUES=(10)
N_VALUES=(500)

COVARIATE_SETS=("default")

RUN_ID=69
REPS=(a b c d e)

# RANDOM=42

# Log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./.log/${RUN_ID}_${SPATIAL_AGGREGATION}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"


source /opt/anaconda3/etc/profile.d/conda.sh  # change to where your anaconda3 lives
conda activate DENV-SEROTYPE-IMPUTATION


for SA in "${SA_VALUES[@]}"; do
   for N in "${N_VALUES[@]}"; do
       for COV in "${COVARIATE_SETS[@]}"; do
           for REP in "${REPS[@]}"; do


               RUN_LABEL="${RUN_ID}${REP}"






               echo "========================================"
               echo "[$(date)] Running combination ID: $RUN_LABEL" | tee -a "$LOG_DIR/find_clusters_tests.log"
               echo "Simulated Annealing Steps: $SA"
               echo "Number of Clustering Runs: $N"
               echo "Covariates: $COV"
               echo "========================================"


               COMPACTNESS=False
               BIOME=False
               KOPPEN=False
               METRO=False
               HUMAN_FOOTPRINT=False
               DENV_100K_CUMULATIVE=False
               DENV_100K_DTW=False
               INDEXP_DTW=False
               SEROTYPES_DTW=False


               if [ "$COV"=="two_covariates" ]; then
                   DENV_100K_DTW=TRUE
                   INDEXP_DTW=TRUE 
               fi


               if [ "$COV"=="default" ]; then
                   COMPACTNESS=TRUE
                   INDEXP_DTW=TRUE
                   HUMAN_FOOTPRINT=TRUE
               fi


               LOG_FILE="$LOG_DIR/run_${RUN_LABEL}_SA${SA}_N${N}_${COV}.log"


               python find-clusters.py \
                   -ID "$RUN_LABEL" \
                   -spatial_aggregation "$SPATIAL_AGGREGATION" \
                   -n "$N" \
                   -max_iterations_sa "$SA" \
                   -threshold 50 \
                   -compactness $COMPACTNESS \
                   -nearest_hypermetro $METRO \
                   -biome $BIOME \
                   -koppen $KOPPEN \
                   -human_footprint $HUMAN_FOOTPRINT \
                   -denv_100k_cumulative $DENV_100K_CUMULATIVE \
                   -denv_100k_DTW $DENV_100K_DTW \
                   -indexP_DTW $INDEXP_DTW \
                   > "$LOG_FILE" 2>&1


           done


           echo "---- Stability Metrics for ID $RUN_ID ----"


python3 - <<EOF
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, mutual_info_score


reps = ["a","b","c","d","e"]
label_sets = []


for rep in reps:
   path = f"../../data/interim/testing_find_clusters_output/${RUN_ID}{rep}/clusters/clusters_rgint.csv"
   df = pd.read_csv(path)
   label_sets.append(df["cluster"].values)


def entropy(labels): # helper func
   _, counts = np.unique(labels, return_counts=True)
   p = counts / counts.sum()
   return -np.sum(p * np.log(p))


def variation_of_information(labels1, labels2):
   Hx = entropy(labels1)
   Hy = entropy(labels2)
   Ixy = mutual_info_score(labels1, labels2)
   return Hx + Hy - 2 * Ixy


aris = []
vis = []


for i in range(5):
   for j in range(i+1, 5): # calculates pairwise ARI and VI
       aris.append(adjusted_rand_score(label_sets[i], label_sets[j]))
       vis.append(variation_of_information(label_sets[i], label_sets[j]))


# output the mean of pairwise ARIs and VIs
print(f"Mean ARI: {np.mean(aris):.4f}")
print(f"Mean VI:  {np.mean(vis):.4f}")


EOF


           ((RUN_ID++))


       done
   done
done


conda deactivate


echo "[$(date)] ✅ All scripts executed successfully!" | tee -a "$LOG_DIR/find_clusters_tests.log"