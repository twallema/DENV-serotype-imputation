import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
from datetime import datetime
import matplotlib.pyplot as plt
from spopt.region import MaxPHeuristic
from libpysal.weights import Rook, Queen
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import numpy as np
from contextlib import redirect_stdout
from scipy.special import softmax

# glasbey color map
from glasbey import create_palette
from matplotlib.colors import ListedColormap

from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# script settings
# >>>>>>>>>>>>>>>

n = 50 # number of max-p regionalization runs to average
threshold = 50  # Sum of column 'N_typed_monthly_mean' should exceed this threshold in every cluster
region_filename = 'rgint' # spatial aggregation: 'mun' (5570 municipalities), 'rgi' (508 immediate regions), 'rgint' (130 intermediate regions)


# helper function
# >>>>>>>>>>>>>>>

def build_co_association_matrix(regions, clusters):
    """
    Build a co-association matrix containing 1 when BR regions belong to the same cluster and 0 if they don't

    input
    -----
    - regions: list
        - list containing unique BR region codes 

    - clusters: list
        - list containing the corresponding ID of the cluster the region belongs to
        - must have the same length as regions

    output
    ------

    - association_matrix: np.ndarray
        - 2D (n x n) association matrix
    """

    # check input length
    assert len(regions) == len(clusters), '`regions` and `clusters` must have the same length'

    # Start with an  matrix of zeros
    n = len(regions)
    association_matrix = np.zeros((n,n),dtype=int) # Maybe rename; this could be confused with the adjacency matrices being created in find-clusters that show which clusters (1-36) are next to each other

    # Loop through each pair of regions and check if they are in the same cluster. Set to 1 if two regions are in the same cluster, 0 if they are not (or the regions are the same)
    for i in range(n):
        for j in range(n):
            if i != j and clusters[i] == clusters[j]:
                association_matrix[i,j] = 1
    return association_matrix



# Load raw data
# >>>>>>>>>>>>>

# Load geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")

# Load case data
denv = pd.read_csv('../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv')
denv['date'] = pd.to_datetime(denv['date'])

# Load cases per 100K data
denv_100k = pd.read_csv(f'../../data/interim/DENV_per_100K/DENV_per_100k_{region_filename}.csv')
denv_100k['date'] = pd.to_datetime(denv_100k['date'])

# Load human footprint 
human_footprint = pd.read_csv(f'../../data/interim/human-footprint/human-footprint_{region_filename}.csv')

# Load DENV per 100K DTW-MDS embedding
DTW_covariates_denv_100k = pd.read_csv(f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-MDS-embedding_{region_filename}.csv')

# Load serotypes DTW-MDS embedding
DTW_covariates_serotypes = pd.read_csv(f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-MDS-embedding_{region_filename}.csv')

# Load indexP DTW-MDS embedding
DTW_covariates_indexP = pd.read_csv(f'../../data/interim/DTW-MDS-embeddings/indexP/DTW-MDS-embedding_{region_filename}.csv')
region = DTW_covariates_indexP.columns.to_list()[0]



# Aggregate incidence and geographical dataset to the intermediate/immediate regions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if region:

    assert ((region != 'CD_RGI') | (region != 'CD_RGINT')), "'region' must be either 'CD_RGI' or 'CD_RGINT''"

    # Geography
    # >>>>>>>>>

    muncipality_region_map = geography[['CD_MUN', f'{region}']]

    # --- 1. Majority vote of biome per immediate region ---
    # Count how many municipalities per biome in each immediate region
    biome_majority = (
        geography.groupby([f'{region}', 'biome'])
        .size()
        .reset_index(name='count')
    )
    # For each immediate region, keep the biome with max count
    biome_majority = (
        biome_majority
        .sort_values([f'{region}', 'count'], ascending=[True, False])
        .drop_duplicates(f'{region}')
        .set_index(f'{region}')['biome']
    )
    # --- 2. Majority vote of Koppen climate per immediate region ---
    # Count how many municipalities per koppen in each immediate region
    koppen_majority = (
        geography.groupby([f'{region}', 'koppen'])
        .size()
        .reset_index(name='count')
    )
    # For each immediate region, keep the koppen with max count
    koppen_majority = (
        koppen_majority
        .sort_values([f'{region}', 'count'], ascending=[True, False])
        .drop_duplicates(f'{region}')
        .set_index(f'{region}')['koppen']
    )
    # --- 3. Dissolve geometries by immediate region ---
    gdf_regions = geography.dissolve(by=f'{region}', aggfunc={'POP': 'sum'})
    # --- 4. Attach the majority biome and koppen back ---
    gdf_regions['biome'] = gdf_regions.index.map(biome_majority)
    gdf_regions['koppen'] = gdf_regions.index.map(koppen_majority)
    # --- 5. Retain only relevant columns ---
    gdf_regions = gdf_regions.reset_index()
    geography = gdf_regions[[f'{region}', 'biome', 'koppen', 'POP', 'geometry']]

    # Incidence
    # >>>>>>>>>

    # Merge incidence with mapping
    denv = denv.merge(muncipality_region_map, on="CD_MUN", how="left")
    # Define custom aggregation function to treat the Nans
    def nan_to_zero_sum(series):
        if series.isna().all():
            return float("nan")
        else:
            return series.fillna(0).sum()
    # List of columns to aggregate
    denv_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]
    # Group and aggregate
    denv = (
        denv.groupby([f"{region}", "date"])[denv_cols]
        .agg(nan_to_zero_sum)
        .reset_index()
    )
   


# Compute threshold
# >>>>>>>>>>>>>>>>>

# Compute the mimimum sum of serotyped cases across all years (will have to be changed)
# limit time window (before 1999 will likely be excluded because it's way too limited; from 2019 onwards all regions have good subtyping)
denv = denv[((denv['date'] > datetime(2000,1,1)) & (denv['date'] < datetime(2008,1,1)))]
# extract year
denv["year"] = pd.to_datetime(denv["date"]).dt.year
# compute total cases per month
denv["N_typed"] = denv[["DENV_1","DENV_2","DENV_3","DENV_4"]].sum(axis=1)
# sum cases by year
active_sum = denv.groupby([f'{region}',"year"])['N_typed'].sum().reset_index()
# take mean across years
mean_active_sum = active_sum.groupby(f'{region}')["N_typed"].mean().reset_index() # array for clustering
mean_active_sum.rename(columns={"N_typed":"N_typed_monthly_mean"}, inplace=True)
# merge min_yearly_sum
geography = geography.merge(mean_active_sum, on=f'{region}', how="left")



# Make biome covariate
# >>>>>>>>>>>>>>>>>>>>

# Make dummies for the biome
biome_dummies = pd.get_dummies(geography["biome"], prefix="biome")
geography = geography.merge(
    biome_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)
# ensure biome dummies are int (0/1)
for col in biome_dummies.columns:
    geography[col] = geography[col].astype(float)



# Make Koppen climate covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Make dummies for the koppen climate
koppen_dummies = pd.get_dummies(geography["koppen"], prefix="koppen")
geography = geography.merge(
    koppen_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)
# Ensure biome dummies are int (0/1)
for col in koppen_dummies.columns:
    geography[col] = geography[col].astype(float)



# Make cumulative DENV per 100K covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# compute cumulative totals
denv_100k = denv_100k.groupby(by=f'{region}')['DENV_per_100k'].sum()
# standardize
sc = StandardScaler()
geography['denv_100k_cumulative'] = sc.fit_transform(denv_100k.values.reshape(-1,1))



# Make human footprint covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# standardize
sc = StandardScaler()
geography['human_footprint'] = sc.fit_transform(human_footprint[["human_footprint"]])



# Make compactness covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>

# 1) Project to Brazil Polyconic (EPSG:5880)
geography = geography.to_crs("EPSG:5880")

# 2) Compute centroids (in metres)
# .centroid is fine after projecting; for very complex multipolygons consider representative_point()
geography["cx"] = geography.geometry.centroid.x
geography["cy"] = geography.geometry.centroid.y

# 3) Standardize and add compactness components
sc = StandardScaler()
geography[["cx","cy"]] = sc.fit_transform(geography[["cx","cy"]])

# 4) Normalize the area codes (similarity in codes reflects proximity in space)
geography[region+'_NORM'] = sc.fit_transform(geography[[region]])



# Make DENV per 100k DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates_denv_100k, 
    on = f'{region}'
)

# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates_denv_100k_names = [x for x in DTW_covariates_denv_100k.columns.to_list() if x != f'{region}']
geography[DTW_covariates_denv_100k_names] = sc.fit_transform(geography[DTW_covariates_denv_100k_names])



# Make indexP DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates_indexP, 
    on = f'{region}'
)

# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates_indexP_names = [x for x in DTW_covariates_indexP.columns.to_list() if x != f'{region}']
geography[DTW_covariates_indexP_names] = sc.fit_transform(geography[DTW_covariates_indexP_names])



# Make serotypes DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates_serotypes, 
    on = f'{region}'
)

# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates_serotypes_names = [x for x in DTW_covariates_serotypes.columns.to_list() if x != f'{region}']
geography[DTW_covariates_serotypes_names] = sc.fit_transform(geography[DTW_covariates_serotypes_names])



# Decide on attributes to use
# >>>>>>>>>>>>>>>>>>>>>>>>>>>

# my pick
attrs = ['cx', 'cy'] + DTW_covariates_indexP_names + ['human_footprint'] #+ koppen_dummies.columns.to_list() # DTW_covariates_denv_100k_names + DTW_covariates_serotypes_names + ['denv_100k_cumulative',] + biome_dummies.columns.to_list()



# Run max-p regionalization model `n` times 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Build contiguity weight map
w = Rook.from_dataframe(geography, use_index=False)

# Setup max-p regionalization model
model = MaxPHeuristic(
    geography,
    w, 
    attrs_name=attrs,
    threshold_name='N_typed_monthly_mean',
    threshold=threshold,
    top_n=3,
    verbose=False,
    policy='multiple',
    max_iterations_construction=100,
    max_iterations_sa=5,
)


n_clusters = []
matrices = []
clusters = pd.DataFrame(index=geography[region].values)

with open("maxp_terminal_log", "w") as f, redirect_stdout(f): # temporarily sends all output to f
    for numRun in range(n):
        print(f"Starting clustering run {numRun+1} of {n}")

        # run model
        model.solve() 

        # append the individual run to dataframes 
        clusters[f'run_{numRun+1}'] = model.labels_
        geography[f'run_{numRun+1}'] = model.labels_

        # save number of clusters
        n_clusters.append(len(np.unique(model.labels_)))

        # save a matrix of size (n_regions x n_regions) containing 1 if regions belong to the same cluster for every run
        matrices.append(build_co_association_matrix(geography[region], model.labels_))



# Extract best_obj_values from f and softmax them
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

best_obj_vals = []
with open ("maxp_terminal_log") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "best objective value:" in line.lower():
        val = float(lines[i+1].strip())
        best_obj_vals.append(val)
weights = softmax(-np.array(best_obj_vals))



# Average co-association matrices across runs
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# compute softmax-weighted mean co-association matrix
for association_matrix, weight in zip(matrices, weights):
   prob_matrix += weight * association_matrix

# save mean co-association matrix
prob_matrix.to_csv(f"../../data/interim/clusters/prob_matrix_{region_filename}.csv")

# compute median number of clusters
n_clusters = int(np.median(n_clusters))

# make a categorical color palette with n_clusters distinct colors
glasbey_cmap = ListedColormap(create_palette(palette_size=n_clusters))



# Randomly select and visualize 12 runs on a 3x4 grid
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# 1. Get all run columns
run_columns = [col for col in clusters.columns if col.startswith("run_")]
# 2. Randomly pick 12 runs
selected_runs = np.random.choice(run_columns, size=12, replace=False)
# 3. Set up the 3x4 grid
fig, axes = plt.subplots(3, 4, figsize=(15, 15))
axes = axes.flatten()
# 4. Plot each run
for ax, run in zip(axes, selected_runs):
    geography.plot(
        column=geography[run],
        categorical=True,
        cmap=glasbey_cmap,
        linewidth=0.2,
        edgecolor="grey",
        legend=False,
        ax=ax
    )
    ax.set_title(run, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f'../../data/interim/clusters/clusters_{region_filename}.png', dpi=300)
plt.close()



# Recluster mean co-association matrix using hierarchical clustering
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Perform hierarchical clustering (average linkage)
Z = linkage(squareform(1 - prob_matrix, checks=False), method='average')

# Choose number of clusters k
geography['consensus_clusters_hierarchical'] = fcluster(Z, n_clusters, criterion='maxclust')



# Recluster mean co-association matrix using hierarchical clustering
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
geography['consensus_clusters_spectral'] = sc.fit_predict(prob_matrix)+1



# Save and visualise the mean clustering results
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# visualise clusters on a map
fig, ax = plt.subplots(nrows=1, ncols=2)
# hierarchical (left)
geography.plot(
    column="consensus_clusters_hierarchical",          # color regions by cluster label
    categorical=True,
    cmap=glasbey_cmap,             # categorical colormap
    linewidth=0.2,
    edgecolor="grey",
    legend=True,
    ax=ax[0],
    legend_kwds={'fontsize': 4, 'ncol': 4, 'loc': 'lower right', 'markerscale': 0.4}
)
ax[0].set_title(f"Hierarchical clustering", fontsize=14)
ax[0].axis("off")
# spectral (right)
geography.plot(
    column="consensus_clusters_spectral",          # color regions by cluster label
    categorical=True,
    cmap=glasbey_cmap,             # categorical colormap
    linewidth=0.2,
    edgecolor="grey",
    legend=False,
    ax=ax[1],
    legend_kwds={'fontsize': 4, 'ncol': 4, 'loc': 'lower right', 'markerscale': 0.4}
)
ax[1].set_title(f"Spectral clustering", fontsize=14)
ax[1].axis("off")
fig.suptitle('Consensus clusters')
plt.tight_layout()
plt.savefig(f'../../data/interim/clusters/consensus_clusters_{region_filename}.png', dpi=300)
plt.close()

# Save the consensus clusters (hierarchical)
clusters = geography[[f'{region}', 'consensus_clusters_hierarchical']]
clusters = clusters.rename(columns={'consensus_clusters_hierarchical': 'cluster'})
clusters.to_csv(f"../../data/interim/clusters/clusters_{region_filename}.csv", index=False)


# Build the clusters' adjacency matrix needed for the Bayesian imputation model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Step 1: Dissolve municipalities to state-level geometries
clusters_geography = geography.dissolve(by='consensus_clusters_hierarchical', as_index=False)
clusters_geography = clusters_geography.reset_index(drop=True)

# Step 2: Ensure 'cluster' column is sorted
clusters_geography = clusters_geography.sort_values('consensus_clusters_hierarchical').reset_index(drop=True)
cluster_list = clusters_geography['consensus_clusters_hierarchical'].tolist()

# Step 4: Build spatial index and adjacency dictionary
sindex = clusters_geography.sindex
adjacency = {idx: set() for idx in cluster_list}

for i, row in clusters_geography.iterrows():
    geom_i = row.geometry
    uf_i = row['consensus_clusters_hierarchical']
    possible_matches_index = list(sindex.intersection(geom_i.bounds))
    
    for j in possible_matches_index:
        if i == j:
            continue
        geom_j = clusters_geography.loc[j, "geometry"]
        uf_j = clusters_geography.loc[j, 'consensus_clusters_hierarchical']
        
        # Use intersects instead of touches for robustness
        if geom_i.intersects(geom_j):
            adjacency[uf_i].add(uf_j)
            adjacency[uf_j].add(uf_i)  # symmetric

# Step 5: Convert to binary adjacency matrix
adj_matrix = pd.DataFrame(0, index=cluster_list, columns=cluster_list)

for uf in cluster_list:
    for neighbor in adjacency[uf]:
        adj_matrix.loc[uf, neighbor] = 1

# Save in a .csv
adj_matrix.to_csv(f'../../data/interim/clusters/adjacency_matrix_{region_filename}.csv')



# Build the clusters' weighted distance matrix for the Bayesian imputation model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Assure appropriate projection
geography = geography.to_crs("EPSG:5880")

# Calculate centroids at f'{region}' level
geography['CENTROID'] = geography.geometry.centroid

# Comptue weighted centroid 
def weighted_centroid(group):
    # Get x and y from centroid
    x = group['CENTROID'].x
    y = group['CENTROID'].y
    weights = group['POP']
    # Weighted average
    x_bar = (x * weights).sum() / weights.sum()
    y_bar = (y * weights).sum() / weights.sum()
    return Point(x_bar, y_bar)

# Group by f'{region}' and calculate weighted centroids
weighted_centroids = geography.groupby('consensus_clusters_hierarchical').apply(weighted_centroid).reset_index()
weighted_centroids.columns = ['consensus_clusters_hierarchical', 'geometry']
centroids_gdf = gpd.GeoDataFrame(weighted_centroids, geometry='geometry', crs=geography.crs)

# Create empty DataFrame
dist_matrix = pd.DataFrame(index=cluster_list, columns=cluster_list, dtype=float)

# Fill with distances in kilometers
for i, row_i in centroids_gdf.iterrows():
    for j, row_j in centroids_gdf.iterrows():
        dist = row_i.geometry.distance(row_j.geometry) / 1000  # meters to km
        dist_matrix.loc[row_i['consensus_clusters_hierarchical'], row_j['consensus_clusters_hierarchical']] = dist

# Save the distance matrix to a csv file
dist_matrix.to_csv(f'../../data/interim/clusters/distance_matrix_{region_filename}.csv')