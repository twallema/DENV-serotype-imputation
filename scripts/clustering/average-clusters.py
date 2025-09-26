import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
import networkx as nx 
import community as community_louvain

# Create basic probability matrix (first an adjacency matrix for each run of find-clusters, and then a probability matrix that gives the probabilities each region will be clustered with every other region)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

# Heatmap of Probability Matrix as it is
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# plt.figure(figsize=(10,8))
# sns.heatmap(prob_matrix, cmap="crest")
# plt.show()

# Clustermap from Hierarchal Clustering, w/o fixed k = 35
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# sns.clustermap(prob_matrix, cmap="viridis", figsize=(12, 12))
# plt.show()

# Manual Hierarchal Clustering from Distance Matrix (1-prob_mat), to fix k clusters -- not sure about this one
'''
dist_matrix = 1 - prob_matrix
np.fill_diagonal(dist_matrix.values, 0) # diagonal still needs to be 0 for squareform to work
dist_matrix.to_csv("../../data/interim/clusters/dist_matrix.csv")

dist_condensed = squareform(dist_matrix.values)

Z = linkage(dist_condensed, method="average") # UPGMA?

labels = fcluster(Z, t=35, criterion="maxclust") # Exactly 35 clusters

clusters = pd.DataFrame({"region": prob_matrix.index, "cluster": labels})
clusters.to_csv("../../data/interim/clusters/hierarchal_consensus_clusters.csv", index=False)

# Reorder prob_matrix by cluster labels
order = np.argsort(labels)
reordered = prob_matrix.values[order][:, order]

# Plot heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(reordered, cmap="viridis", cbar=True)
plt.title("Consensus Clusters (Hierarchical, k=35)")
# plt.show()
'''

# Spectral Clustering
# >>>>>>>>>>>>>>>>>>>>>>>>>>
clustering = SpectralClustering(affinity='precomputed', random_state=0)
labels = clustering.fit_predict(prob_matrix.values)

# Reorder matrix by cluster assignment in order to see groupings
sorted_spec = np.argsort(labels)
sorted_regions = prob_matrix.index[sorted_spec]

sorted_mat = prob_matrix.loc[sorted_regions, sorted_regions]

# sns.heatmap(sorted_mat, cmap="viridis")
# plt.show()

# Community Detection - Louvain
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
G = nx.from_pandas_adjacency(prob_matrix)

partition = community_louvain.best_partition(G,weight='weight',resolution=17)

cd_clusters = pd.DataFrame({
    "region": list(partition.keys()),
    "cluster": list(partition.values())
})

cd_clusters.to_csv("../../data/interim/clusters/louvain_clusters.csv", index=False)

pos = nx.spring_layout(G, seed=42) 
plt.figure(figsize=(8,8))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=list(partition.values()), cmap=plt.cm.get_cmap("tab20",40))
nx.draw_networkx_edges(G, pos, alpha=0.05)
plt.axis("off")
plt.show()


# New probability matrix with weighted averages based on the Best Objective Value score from each run (20 runs for testing rather than 50)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
