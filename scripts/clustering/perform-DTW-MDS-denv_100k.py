
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from sklearn.manifold import MDS
from pygam import LinearGAM, s
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# glasbey color map
from glasbey import create_palette
from matplotlib.colors import ListedColormap

# spatial aggregation: 'mun' (5570 municipalities), 'rgi' (508 immediate regions), 'rgint' (130 intermediate regions)
region_filename = 'rgi'
region = 'CD_RGI'
# dynamic time warping
sakoe_chiba_radius = 12  # 0 will emphasise similarities in seasonality, 12/24/36 emphasizes similarity in magnitudes
# number of dimensions to project the DTW matrix onto (bigger = better representation of DTW matrix BUT clustering becomes harder)
n_mds_components = 10
# use all data
start_date = datetime(1000, 1, 1)
end_date = datetime(3000, 1, 1)


# --- Step 1: Prepare and smooth incidence time series ---

# get the data
denv = pd.read_csv(f'../../data/interim/DENV_per_100K/DENV_per_100k_{region_filename}.csv')
denv['date'] = pd.to_datetime(denv['date'])

# cut from startdate
denv = denv[((denv['date'] >= start_date) & (denv['date'] <= end_date))]

# GAM smooth log1p timeseries
denv["date_num"] = (denv["date"] - denv["date"].min()).dt.days
denv["DENV_per_100k_log1p"] = np.log1p(denv["DENV_per_100k"])

denv["pred_DENV_per_100k_log1p"] = np.nan
for cd_rgi, group in denv.groupby(f"{region}"):

    X = group[["date_num"]].values
    y = group["DENV_per_100k_log1p"].values

    gam = LinearGAM(s(0, n_splines=27*5, lam=0.1)).fit(X, y)

    denv.loc[group.index, "pred_DENV_per_100k_log1p"] = gam.predict(X)

denv["pred_DENV_per_100k"] = np.expm1(denv["pred_DENV_per_100k_log1p"])

# # visualise results
# fig,ax=plt.subplots()
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 530001]['pred_DENV_per_100k_log1p'], color='black', label='530001')
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 110001]['pred_DENV_per_100k_log1p'], color='red', label='110001')
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 110002]['pred_DENV_per_100k_log1p'], color='green', label='110002')
# ax.legend()
# plt.show()
# plt.close()

# import sys
# sys.exit()

# --- Step 2: Dynamic time warping ---

# pivot to wide format
ts = denv.pivot(index=f'{region}', columns='date', values='pred_DENV_per_100k_log1p')

# tslearn expects 3D array: (n_ts, n_timesteps, 1)
X = ts.fillna(0).to_numpy()[:, :, np.newaxis]

# compute pairwise DTW distances
dtw_dist = cdist_dtw(X, sakoe_chiba_radius=0, verbose=False)

# visualise raw matrx
plt.figure(figsize=(10, 8))
plt.imshow(dtw_dist, cmap="viridis", aspect="auto")
plt.colorbar(label="DTW distance")
plt.title(f"DTW distance matrix")
plt.axis("off")  # hide axis labels since 508 is too dense
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-mat-raw_{region_filename}.png', dpi=600)
plt.close()

# visualise clustermap
sns.clustermap(dtw_dist, cmap="viridis", figsize=(12, 12))
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-mat-clustermap_{region_filename}.png', dpi=600)
plt.close()


# --- Step 3: Cluster DTW matrix and visualise on a map ---

# Load geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
# Dissolve to states
gdf_states = geography.dissolve(by='CD_UF')
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
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-mat-clustered_{region_filename}.png', dpi=400)
plt.close()


# --- Step 4: Visualise our favorite on a map and export to svg ---

# Visualise DTW clusters 
fig,ax=plt.subplots()
gdf_states.boundary.plot(ax=ax, linewidth=0.5, color="black")
geography.boundary.plot(ax=ax, linewidth=0.1, alpha=0.3, color="black")
geography.plot(
    column=f"dtw_clusters_7",          # color regions by cluster label
    categorical=True,
    linewidth=0,
    edgecolor=None,
    legend=False,
    ax=ax,
)
ax.axis("off")

plt.tight_layout()
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-mat-favorite_{region_filename}.svg')
plt.close()

# --- Step 5: Multidimensional Scaling (MDS) ---

# perform MDS
mds = MDS(n_components=n_mds_components, dissimilarity="precomputed", random_state=42, max_iter=10000, normalized_stress=True)
coords = mds.fit_transform(dtw_dist)
# evaluate performance metric (0.025=excellent, 0.05=good, 0.10=fair, 0.20=poor)
print(mds.stress_)
# convert to dataframe
embedding = pd.DataFrame(coords, index=ts.index, columns=[f"denv_100k_mds{i+1}" for i in range(n_mds_components)]).reset_index()
# save dataframe
embedding.to_csv(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-MDS-embedding_{region_filename}.csv', index=False)