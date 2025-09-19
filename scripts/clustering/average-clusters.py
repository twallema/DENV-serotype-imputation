# Creates an adjacency matrix for each run of find-clusters, and then a probability matrix that gives the probabilities each region will be clustered with every other region

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def build_adjacency(df):
    # Convert the two columns to arrays
    regions = df["CD_RGINT"].to_numpy()
    clusters = df["cluster"].to_numpy()

    # Start with a matrix of zeros
    n = len(regions)
    adj_matrix = np.zeros((n,n),dtype=int) # Maybe rename; this could be confused with the adjacency matrices being created in find-clusters that show which clusters (1-36) are next to each other

    # Loop through each pair of regions and check if they are in the same cluster. Set to 1 if two regions are in the same cluster, 0 if they are not (or the regions are the same)
    for i in range(n):
        for j in range(n):
            if i != j and clusters[i] == clusters[j]:
                adj_matrix[i,j] = 1

    adj_df = pd.DataFrame(adj_matrix, index=regions, columns=regions)
    return adj_df

numRuns = 50 # Make sure this matches numRuns in find-clusters.py
adj_matrices = []

for run in range(1, numRuns+1):
    # Load dataset of intermediate regions and the clusters they belong to
    df = pd.read_csv(f'../../data/interim/clusters/clusters_rgint_run{run}.csv')
    adj_df = build_adjacency(df)
    adj_matrices.append(adj_df)

# Probability matrix averaging runs
regions = adj_matrices[0].index 
prob_matrix = pd.DataFrame(0.0, index=regions, columns=regions)

for adj in adj_matrices:
    prob_matrix += adj

prob_matrix /= numRuns

prob_matrix.to_csv("../../data/interim/clusters/prob_matrix_test.csv")

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(prob_matrix, cmap="crest")
# plt.show()

sns.clustermap(prob_matrix, cmap="viridis", figsize=(12, 12))
# plt.savefig(f'../../data/interim/clusters/prob_matrix_clustermap{region_filename}.pdf')
plt.show()
# plt.close()