
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pygam import LinearGAM, s


# glasbey color map
from glasbey import create_palette
from matplotlib.colors import ListedColormap


# spatial aggregation: 'mun' (5570 municipalities), 'rgi' (508 immediate regions), 'rgint' (130 intermediate regions)
region_filename = 'rgi'
region = 'CD_RGI'
# dynamic time warping
sakoe_chiba_radius = 0
# number of dimensions to project the DTW matrix onto (bigger = better representation of DTW matrix BUT clustering becomes harder)
n_mds_components = 3


# --- Step 1: Prepare and smooth incidence time series ---

# get the index P data
indexP = pd.read_csv(f'../../data/interim/indexP/indexP_{region_filename}.csv')

# GAM smooth log1p timeseries
indexP["month_num"] = indexP["month"] - indexP["month"].min()

indexP["indexP_smooth"] = np.nan
for _, group in indexP.groupby(f"{region}"):

    X = group[["month_num"]].values
    y = group["indexP"].values

    gam = LinearGAM(s(0), n_splines=10, lam=0.2).fit(X, y)

    indexP.loc[group.index, "indexP_smooth"] = gam.predict(X)


# # visualise results
# fig,ax=plt.subplots()
# ax.plot(indexP.month.unique(), indexP[indexP['CD_RGI'] == 530001]['indexP'], color='black', label='530001')
# ax.plot(indexP.month.unique(), indexP[indexP['CD_RGI'] == 530001]['indexP_smooth'], color='red', label='530001 (smooth)')
# ax.plot(indexP.month.unique(), indexP[indexP['CD_RGI'] == 110001]['indexP'], color='black', linestyle='--', label='110001')
# ax.plot(indexP.month.unique(), indexP[indexP['CD_RGI'] == 110001]['indexP_smooth'], color='red', linestyle='--', label='110001 (smooth)')
# ax.legend()
# plt.show()
# plt.close()


# --- Step 2: Compute DTW distance matrix ---

# pivot to wide format
ts = indexP.pivot(index=f'{region}', columns='month', values='indexP_smooth')

# tslearn expects 3D array: (n_ts, n_timesteps, 1)
X = ts.fillna(0).to_numpy()[:, :, np.newaxis]

# compute pairwise DTW distances
dtw_dist = cdist_dtw(X, sakoe_chiba_radius=sakoe_chiba_radius, n_jobs=-1, verbose=True)

# visualise raw matrx
plt.figure(figsize=(10, 8))
plt.imshow(dtw_dist, cmap="viridis", aspect="auto")
plt.colorbar(label="DTW distance")
plt.title(f"DTW distance matrix")
plt.axis("off")  # hide axis labels since 508 is too dense
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/indexP/DTW-mat-raw_{region_filename}.pdf')
plt.close()

# visualise clustermap
sns.clustermap(dtw_dist, cmap="viridis", figsize=(12, 12))
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/indexP/DTW-mat-clustermap_{region_filename}.pdf')


# --- Step 3: Cluster DTW matrix and visualise on a map ---

# Load geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
# Dissolve to desired spatial level
geography = geography.dissolve(by=f'{region}', aggfunc={'POP': 'sum'})
# Perform hierarchical clustering (average linkage)
n_clusters_list = [2, 3, 4, 5, 6, 7, 10]
for n_clusters in n_clusters_list:  
    # Hierarchical clustering
    Z = linkage(squareform(dtw_dist), method='average')
    geography[f'dtw_clusters_{n_clusters}'] = fcluster(Z, n_clusters, criterion='maxclust')

# Visualise DTW clusters 
fig,ax=plt.subplots(ncols=len(n_clusters_list))

for i,n_clusters in enumerate(n_clusters_list):
    glasbey_cmap = ListedColormap(create_palette(palette_size=n_clusters))
    geography.plot(
        column=f"dtw_clusters_{n_clusters}",          # color regions by cluster label
        categorical=True,
        cmap=glasbey_cmap,             # categorical colormap
        linewidth=0,
        edgecolor=None,
        legend=False,
        ax=ax[i],
        legend_kwds={'fontsize': 7, 'ncol': 2, 'loc': 'lower right'}
    )
    ax[i].axis("off")
    ax[i].set_title(f'{n_clusters} clusters')
plt.tight_layout()
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/indexP/DTW-mat-clustered_{region_filename}.png', dpi=400)
plt.close()


# --- Step 4: Multidimensional Scaling (MDS) ---

# perform MDS
mds = MDS(n_components=n_mds_components, dissimilarity="precomputed", random_state=42, max_iter=10000, normalized_stress=True)
coords = mds.fit_transform(dtw_dist)
# evaluate performance metric (0.025=excellent, 0.05=good, 0.10=fair, 0.20=poor)
print(mds.stress_)
# convert to dataframe
embedding = pd.DataFrame(coords, index=ts.index, columns=[f"indexP_mds{i+1}" for i in range(n_mds_components)]).reset_index()
# save dataframe
embedding.to_csv(f'../../data/interim/DTW-MDS-embeddings/indexP/DTW-MDS-embedding_{region_filename}.csv', index=False)