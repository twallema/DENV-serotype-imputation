
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from sklearn.manifold import MDS

# glasbey color map
from glasbey import create_palette
from matplotlib.colors import ListedColormap

###########
## Notes ##
###########

# This script attempts to perform a DTW on the smoothed ratio of DENV 2 between the years 2018-2024
# This period is chosen because the number of collected samples in this period is much higher than before & therefore we might be able to levarage the DTW distance between the prevalent serotypes might inform better clusters
# However, my opinion is this procedure only works for DENV 2 (most prevalent) and on the level of the intermediate regions; finer spatial resolution is sadly not attainable


# spatial aggregation: 'rgint' (130 intermediate regions) ONLY
region_filename = 'rgint'   # NOTE: script intended to work with 'rgint'
region = 'CD_RGINT'         # NOTE: script intended to work with 'CD_RGINT'
# number of dimensions to project the DTW matrix onto (bigger = better representation of DTW matrix BUT clustering becomes harder)
n_mds_components = 3        # NOTE: script intended to work with 3
# use all data
start_date = datetime(2022, 7, 1)
end_date = datetime(2025, 7, 1)


# --- Step 1: Prepare and smooth incidence time series ---

# Load case data
denv = pd.read_csv('../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv')
denv['date'] = pd.to_datetime(denv['date'])

# cut from startdate to enddate
denv = denv[((denv['date'] >= start_date) & (denv['date'] <= end_date))]

# aggregate data spatially
# get mapping
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
muncipality_region_map = geography[['CD_MUN', f'{region}']]
# merge the RGI mapping into the main dataframe
denv = denv.merge(muncipality_region_map, on="CD_MUN", how="left")
# define a custom aggregation function to treat the Nans
def sum_with_nan(series):
    if series.isna().all():
        return np.nan
    else:
        return series.fillna(0).sum()
# aggregate by CD_RGI and date
denv = denv.groupby([f'{region}', "date"], as_index=False)[["DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]].agg(sum_with_nan)

# compute total serotyped cases
def sum_serotypes(row):
    if row[["DENV_1", "DENV_2", "DENV_3", "DENV_4"]].isna().all():
        return np.nan
    else:
        return row[["DENV_1", "DENV_2", "DENV_3", "DENV_4"]].fillna(0).sum()
# Apply row-wise
denv["total_serotyped"] = denv.apply(sum_serotypes, axis=1)
# compute fractions of DENV_x
denv['f1'] = denv['DENV_1'] / denv['total_serotyped']
denv['f2'] = denv['DENV_2'] / denv['total_serotyped']

# sort dataframe
denv = denv.sort_values([f'{region}', 'date'])

# interpolate NaNs linearly per CD_RGI
denv[["f1", "f2"]] = denv.groupby(f'{region}')[["f1", "f2"]].transform(
    lambda group: group.interpolate(method="linear", limit_direction="both")
)

# then fit a spline
from scipy.interpolate import UnivariateSpline
def smooth_series(series):
    x = np.arange(len(series))
    mask = ~np.isnan(series)
    x_valid = x[mask]
    y_valid = series[mask]

    # If too few points, just return original or linear fill
    if len(x_valid) < 2:  # cannot fit spline
        print("There's too few")
        return series  # or series.fillna(method='ffill').fillna(method='bfill')

    # Fit spline to valid points
    spline = UnivariateSpline(x_valid, y_valid, s=1)
    return np.clip(spline(x), 0, 1)


# apply per f'{region}'
for col in ["f1", "f2"]:
    denv[col + "_smooth"] = denv.groupby(f'{region}')[col].transform(
        lambda s: smooth_series(s.values)
    )

# visualise results
fig,ax=plt.subplots()
ax.plot(denv.date.unique(), denv[denv['CD_RGINT'] == 5301]['f2'], color='red', label='530001')
ax.plot(denv.date.unique(), denv[denv['CD_RGINT'] == 5301]['f2_smooth'], color='red', linestyle='--', label='530001 (smooth)')
ax.plot(denv.date.unique(), denv[denv['CD_RGINT'] == 1101]['f2'], color='black', label='110001')
ax.plot(denv.date.unique(), denv[denv['CD_RGINT'] == 1101]['f2_smooth'], color='black', linestyle='--', label='110001 (smooth)')
ax.legend()
plt.show()
plt.close()


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
    for denv_strain in ['1', '2']:

        print(f"DTW on season '{season}' for denv strain {denv_strain}")

        # slice data
        denv_season = denv[denv['season'] == season]

        # pivot to wide format
        ts = denv_season.pivot(index=f'{region}', columns='date', values=f'f{denv_strain}_smooth')

        # tslearn expects 3D array: (n_ts, n_timesteps, 1)
        X = ts.fillna(0).to_numpy()[:, :, np.newaxis]

        # compute pairwise DTW distances
        dtw_dist = cdist_dtw(X, sakoe_chiba_radius=1, n_jobs=-1, verbose=False)

        # append to output
        dtw_dist_save.append(dtw_dist)

# make a tensor & average out season and dengue strain axis
dtw_dist = np.mean(np.stack(dtw_dist_save, axis=-1), axis=-1)


# visualise raw matrx
plt.figure(figsize=(10, 8))
plt.imshow(dtw_dist, cmap="viridis", aspect="auto")
plt.colorbar(label="DTW distance")
plt.title(f"DTW distance matrix")
plt.axis("off")  # hide axis labels since 508 is too dense
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-mat-raw_{region_filename}.pdf')
plt.close()

# visualise clustermap
sns.clustermap(dtw_dist, cmap="viridis", figsize=(12, 12))
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-mat-clustermap_{region_filename}.pdf')


# --- Step 4: Cluster DTW matrix and visualise on a map ---

from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Load geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
# Dissolve to desired spatial level
geography = geography.dissolve(by=f'{region}', aggfunc={'POP': 'sum'})
# Perform hierarchical clustering (average linkage)
n_clusters_list = [3, 4, 5, 10, 15]
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
plt.suptitle('DTW on serotype ratios\n(Mean of DENV 2 ratio, season-to-season from 2022-2025)')
plt.tight_layout()
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-mat-clustered_{region_filename}.png', dpi=400)
plt.close()


# --- Step 5: Multidimensional Scaling (MDS) ---

# perform MDS
mds = MDS(n_components=n_mds_components, dissimilarity="precomputed", random_state=42, max_iter=10000, normalized_stress=True)
coords = mds.fit_transform(dtw_dist)
# evaluate performance metric (0.025=excellent, 0.05=good, 0.10=fair, 0.20=poor)
print(mds.stress_)
# convert to dataframe
embedding = pd.DataFrame(coords, index=ts.index, columns=[f"serotypes_mds{i+1}" for i in range(n_mds_components)]).reset_index()
# save dataframe
embedding.to_csv(f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-MDS-embedding_{region_filename}.csv', index=False)

# --- Step 6: Upsample spatially from RGINT to RGI and MUN

# NOTE: imparts the shape of the intermediate regions onto the clustering which doesn't seem appropriate to me
# Load geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
# Extract mapping
muncipality_region_map = geography[['CD_MUN', 'CD_RGI', 'CD_RGINT']]
# Merge mapping to embeddings
embedding = embedding.merge(muncipality_region_map, on="CD_RGINT", how="left")
# Simply take the desired columns for the municipalities
embedding[['CD_MUN', 'serotypes_mds1', 'serotypes_mds2', 'serotypes_mds3']].to_csv(f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-MDS-embedding_mun.csv', index=False)
# Filter out unique values
embedding[['CD_RGI', 'serotypes_mds1', 'serotypes_mds2', 'serotypes_mds3']].groupby(by='CD_RGI').first().reset_index().to_csv(f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-MDS-embedding_rgi.csv', index=False)
