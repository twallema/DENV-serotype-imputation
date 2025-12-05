import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score


def computeMetrics(run, reps):
    labelSets = []

    for rep in reps:
        path = f"../../data/interim/testing_find_clusters_output/{run}{rep}/clusters/clusters_rgint.csv"

        df = pd.read_csv(path)
        labels = df["cluster"].values
        labelSets.append(labels)
    
    aris = []
    vis = []
    for i in range(4):
        for j in range(i + 1, 4):
            ari = adjusted_rand_score(labelSets[i], labelSets[j])
            vi = variation_of_information(labelSets[i], labelSets[j])
            aris.append(ari)
            vis.append(vi)
        
    meanARI = float(np.mean(aris))
    meanVI = float(np.mean(vis))

    return meanARI, meanVI, labelSets

def heatmapARI(labelSets, reps, run):
    l = len(labelSets)
    mat = np.zeros((l,l))

    for i in range(l):
        for j in range(l):
            mat[i, j] = adjusted_rand_score(labelSets[i], labelSets[j])

    df = pd.DataFrame(mat, index=reps, columns=reps)

    plt.figure(figsize=(5, 5))
    sns.heatmap(df, annot=True, cmap="viridis", vmin=0, vmax=1)
    plt.title(f"ARI Heatmap: Run {run}")
    plt.savefig(f"../../data/interim/testing_find_clusters_output/ari_heatmap_run_{run}.png", dpi=300)
    plt.close()

def variation_of_information(labels1, labels2): # Chat help
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log(p))

    Hx = entropy(labels1)
    Hy = entropy(labels2)
    Ixy = mutual_info_score(labels1, labels2)

    return Hx + Hy - 2 * Ixy



runs = range(62, 63)
reps = ["a", "b", "c", "d"]

print("\n==============================")
print("             Metrics")
print("==============================\n")

for run in runs:

    meanARI, meanVI, labelSets = computeMetrics(run, reps)
    print(f"Run {run}: mean ARI = {meanARI:.4f}, mean VI = {meanVI:.4f}")

    heatmapARI(labelSets, reps, run)

print("\n==============================")
print("            DONE")
print("==============================\n")