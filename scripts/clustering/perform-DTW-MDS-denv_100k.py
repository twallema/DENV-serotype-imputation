
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tslearn.metrics import cdist_dtw
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# spatial aggregation: 'mun' (5570 municipalities), 'rgi' (508 immediate regions), 'rgint' (130 intermediate regions)
region_filename = 'mun'
region = 'CD_MUN'
# number of dimensions to project the DTW matrix onto (bigger = better representation of DTW matrix BUT clustering becomes harder)
n_mds_components = 4
# sigma of gaussian filter used to smooth DENV incidence per 100K
sigma = 1
# z-score the DENV incidence per 100K (doesn't work well; just here to let you know I tried this)
z_score = False
# use all data
start_date = datetime(1000, 1, 1)
end_date = datetime(3000, 1, 1)


# --- Step 1: Prepare and smooth incidence time series ---

# get the data
denv = pd.read_csv(f'../../data/interim/DENV_per_100K/DENV_per_100k_{region_filename}.csv')
denv['date'] = pd.to_datetime(denv['date'])

# cut from startdate
denv = denv[((denv['date'] >= start_date) & (denv['date'] <= end_date))]

# define gaussian smoother
def smooth_gaussian(x, sigma):
    values = x.fillna(0).to_numpy()
    return gaussian_filter1d(values, sigma=sigma)

# perform smoothing with guassian filter
denv['DENV_per_100k_smooth'] = (
    denv.groupby(f'{region}')['DENV_per_100k']
      .transform(lambda x: smooth_gaussian(x, sigma=sigma))
)

# Z-score
if z_score:
    def zscore(x):
        return (x - x.mean()) / x.std(ddof=0)
    denv["DENV_per_100k_smooth"] = (
        denv.groupby(f'{region}')["DENV_per_100k_smooth"]
        .transform(zscore)
    )

# # visualise results
# fig,ax=plt.subplots()
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 530001]['DENV_per_100k'], color='black', label='530001')
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 530001]['DENV_per_100k_smooth'], color='red', label='530001 (smooth)')
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 110001]['DENV_per_100k'], color='black', linestyle='--', label='110001')
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 110001]['DENV_per_100k_smooth'], color='red', linestyle='--', label='110001 (smooth)')
# ax.legend()
# plt.show()
# plt.close()


# --- Step 2: Attach a 'season' label

def assign_season(df, split_month=7):
    """
    Assigns a season label to each row of a dataframe based on a split month.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing a 'date' column (datetime64 dtype).
    split_month : int
        Month (1–12) where the seasonal split occurs. 
        Example: 7 means July -> season runs from July to next June.
    
    Returns:
    --------
    pd.Series
        A column of season labels in the format 'YYYY-YYYY'.
    """
    year = df['date'].dt.year
    month = df['date'].dt.month

    # If before the split month, assign season as (year-1)-(year)
    # If on/after split month, assign season as (year)-(year+1)
    season_start = (year - 1).where(month < split_month, year)
    season_end = season_start + 1
    
    return season_start.astype(str) + "-" + season_end.astype(str)


# Append seasons
denv['season'] = assign_season(denv, split_month=7)


# --- Step 3: Loop over seasons & Compute DTW distance matrix ---

dtw_dist_save = []
for season in denv['season'].unique():

    print(f"DTW on season '{season}'")

    # slice data
    denv_season = denv[denv['season'] == season]

    # pivot to wide format
    ts = denv_season.pivot(index=f'{region}', columns='date', values='DENV_per_100k_smooth')

    # tslearn expects 3D array: (n_ts, n_timesteps, 1)
    X = ts.fillna(0).to_numpy()[:, :, np.newaxis]

    # compute pairwise DTW distances
    dtw_dist = cdist_dtw(X, sakoe_chiba_radius=1, n_jobs=-1, verbose=False)

    # append to output
    dtw_dist_save.append(dtw_dist)

# make a tensor & average out season axis
dtw_dist = np.mean(np.stack(dtw_dist_save, axis=-1), axis=-1)


# visualise raw matrx
plt.figure(figsize=(10, 8))
plt.imshow(dtw_dist, cmap="viridis", aspect="auto")
plt.colorbar(label="DTW distance")
plt.title(f"DTW distance matrix")
plt.axis("off")  # hide axis labels since 508 is too dense
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-mat-raw_{region_filename}.pdf')
plt.close()

# visualise clustermap
sns.clustermap(dtw_dist, cmap="viridis", figsize=(12, 12))
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-mat-clustermap_{region_filename}.pdf')


# --- Step 4: Cluster DTW matrix and visualise on a map ---

from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Load geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
# Dissolve to desired spatial level
geography = geography.dissolve(by=f'{region}', aggfunc={'POP': 'sum'})
# Perform hierarchical clustering (average linkage)
n_clusters_list = [5, 10, 15, 25, 50, 100]
for n_clusters in n_clusters_list:  
    # Hierarchical clustering
    Z = linkage(squareform(dtw_dist), method='average')
    geography[f'dtw_clusters_{n_clusters}'] = fcluster(Z, n_clusters, criterion='maxclust')
    # Spectral clustering
    #sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    #geography[f'dtw_clusters_{n_clusters}'] = sc.fit_predict(dtw_dist)

# Visualise DTW clusters 
fig,ax=plt.subplots(ncols=len(n_clusters_list))

for i,n_clusters in enumerate(n_clusters_list):
    geography.plot(
        column=f"dtw_clusters_{n_clusters}",          # color regions by cluster label
        categorical=True,
        cmap="tab20",             # categorical colormap
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