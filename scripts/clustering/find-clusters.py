import io, sys, os
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
from glasbey import create_palette
from matplotlib.colors import ListedColormap
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import argparse

# parse arguments
# >>>>>>>>>>>>>>>

# example use:
# $ python find-clusters.py -ID test -spatial_aggregation rgint

# helper function for argument parsing
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("-ID", type=str, help="Identifier of the pipeline run.")
parser.add_argument("-n", type=int, help="Number of clustering runs to average.", default=250)
parser.add_argument("-max_iterations_sa", type=int, help="Number of simulated annealing steps.", default=10)
parser.add_argument("-threshold", type=float, help="Minimal number of serotyped cases in a cluster.", default=75)
parser.add_argument("-spatial_aggregation", type=str, help="Spatial aggregation clustering was performed on.")
parser.add_argument("-validation_bw", type=float, help="Fraction of spatial units left out for within-sample validation.", default=0)
# covariates
parser.add_argument("-compactness", type=str_to_bool, help="Include cluster compactness as a covariate in clustering.", default=True)
parser.add_argument("-nearest_hypermetro", type=str_to_bool, help="Include nearest hypermetro area as a covariate in clustering.", default=True)
parser.add_argument("-biome", type=str_to_bool, help="Include biome covariate in clustering.", default=True)
parser.add_argument("-koppen", type=str_to_bool, help="Include Koppen climate classification covariate in clustering.", default=False)
parser.add_argument("-human_footprint", type=str_to_bool, help="Include human footprint classification covariate in clustering.", default=True)
parser.add_argument("-denv_100k_cumulative", type=str_to_bool, help="Include confirmed cumulative DENV incidence per 100K as a covariate in clustering.", default=False)
parser.add_argument("-denv_100k_DTW", type=str_to_bool, help="Include DTW of confirmed cumulative DENV incidence per 100K as a covariate in clustering.", default=False)
parser.add_argument("-indexP_DTW", type=str_to_bool, help="Include DTW of index P as a covariate in clustering.", default=True)
parser.add_argument("-serotypes_DTW", type=str_to_bool, help="Include DTW of recent (2020-2025) serotyped cases as a covariate in clustering.", default=False)
args = parser.parse_args()

# assign to desired variables
ID = args.ID
n = args.n
max_iterations_sa = args.max_iterations_sa
threshold = args.threshold
spatial_aggregation = args.spatial_aggregation
validation_bw = args.validation_bw
include_biome = args.biome
include_koppen = args.koppen
include_human_footprint = args.human_footprint
include_compactness = args.compactness
include_nearest_hypermetro = args.nearest_hypermetro
include_denv_100k_cumulative = args.denv_100k_cumulative
include_denv_100k_DTW = args.denv_100k_DTW
include_indexP_DTW = args.indexP_DTW
include_serotypes_DTW = args.serotypes_DTW


# pipeline output folder
abs_dir = os.path.dirname(__file__) # make sure all referenced paths are relative to the location of this file and not the terminal's pwd
output_folder = os.path.join(abs_dir, f'../../data/interim/testing_find_clusters_output/{ID}/clusters/') # changed to testing location
# check if output dir exists, if not, make it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
geography = gpd.read_parquet(os.path.join(abs_dir, "../../data/interim/geographic-dataset.parquet"))

# Load case data
denv = pd.read_csv(os.path.join(abs_dir,'../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv'))
denv['date'] = pd.to_datetime(denv['date'])

# Load cases per 100K data
denv_100k = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DENV_per_100K/DENV_per_100k_{spatial_aggregation}.csv'))
denv_100k['date'] = pd.to_datetime(denv_100k['date'])

# Load human footprint 
human_footprint = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/human-footprint/human-footprint_{spatial_aggregation}.csv'))

# Load DENV per 100K DTW-MDS embedding
DTW_covariates_denv_100k = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-MDS-embedding_{spatial_aggregation}.csv'))

# Load serotypes DTW-MDS embedding
DTW_covariates_serotypes = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/serotypes/DTW-MDS-embedding_{spatial_aggregation}.csv'))

# Load indexP DTW-MDS embedding
DTW_covariates_indexP = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/indexP/DTW-MDS-embedding_{spatial_aggregation}.csv'))
region = DTW_covariates_indexP.columns.to_list()[0]

# Load nearest hypermetro area
nearest_hypermetro = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/nearest-hypermetro/nearest-hypermetro_{spatial_aggregation}.csv'))


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
   


# Compute threshold & leave out within-sample validation
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Compute the mimimum sum of serotyped cases across all years (will have to be changed)
# limit time window (before 1999 will likely be excluded because it's way too limited; from 2019 onwards all regions have good subtyping)
denv = denv[((denv['date'] > datetime(2000,1,1)) & (denv['date'] < datetime(2019,1,1)))]
# extract year
denv["year"] = pd.to_datetime(denv["date"]).dt.year
# compute total cases per month
denv["N_typed"] = denv[["DENV_1","DENV_2","DENV_3","DENV_4"]].sum(axis=1)
# sum cases by year
yearly_sum = denv.groupby([f'{region}',"year"])['N_typed'].sum().reset_index()
# take mean across years
yearly_sum_median = yearly_sum.groupby(f'{region}')["N_typed"].mean() # array for clustering
yearly_sum_median.rename("N_typed_yearly_median", inplace=True)

# perform training-validation split
## Take validation_bw around Q1 territories
validation_center = 0.25
validation_labels = yearly_sum_median.loc[((yearly_sum_median > np.quantile(yearly_sum_median, q=validation_center-validation_bw)) & (yearly_sum_median < np.quantile(yearly_sum_median, q=validation_center+validation_bw)))].index.values.tolist()
## Take validation_bw around Q2 territories
validation_center = 0.50
validation_labels.extend(yearly_sum_median.loc[((yearly_sum_median > np.quantile(yearly_sum_median, q=validation_center-validation_bw)) & (yearly_sum_median < np.quantile(yearly_sum_median, q=validation_center+validation_bw)))].index.values.tolist())
yearly_sum_median.loc[validation_labels] = 0

# convert validation labels to municipality level (because the incidence data used in the bayesian model is too)
validation_labels_muni = muncipality_region_map[muncipality_region_map[f'{region}'].isin(validation_labels)]['CD_MUN']
validation_labels_muni.to_csv(os.path.join(output_folder, 'validation_labels.csv'), index=False)

# visualise where the left out areas are
geography['validation_labels'] = geography[f'{region}'].isin(validation_labels)
fig,ax=plt.subplots()
geography.plot(
    column='validation_labels',
    categorical=True,
    linewidth=0.2,
    edgecolor="grey",
    legend=False,
    ax=ax
)
ax.set_title('Areas left out during within-sample validation', fontsize=10)
ax.axis("off")
plt.savefig(os.path.join(output_folder, 'validation_labels.png'), dpi=300)
plt.close()


# merge min_yearly_sum
geography = geography.merge(yearly_sum_median, on=f'{region}', how="left")



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
# make a covariate name mapping
covariate_names = []
if include_biome:
    covariate_names.extend(biome_dummies.columns.to_list())



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
# add to covariate name mapping
if include_koppen:
    covariate_names.extend(koppen_dummies.columns.to_list())


# Make nearest hypermetro area covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Make dummies for the nearest hypermetro covariate
nearest_hypermetro_dummies = pd.get_dummies(nearest_hypermetro['hypermetro_id'], prefix="nearest_hypermetro")
# Merge to the geography dataframe
geography = geography.merge(
    nearest_hypermetro_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)
# Ensure nearest_hypermetro dummies are int (0/1)
for col in nearest_hypermetro_dummies.columns:
    geography[col] = geography[col].astype(float)
# add to covariate name mapping
if include_nearest_hypermetro:
    covariate_names.extend(nearest_hypermetro_dummies.columns.to_list())


# Make human footprint covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# standardize
sc = StandardScaler()
geography['human_footprint'] = sc.fit_transform(human_footprint[["human_footprint"]])
# add to covariate name mapping
if include_human_footprint:
    covariate_names.extend(['human_footprint',])



# Make cumulative DENV per 100K covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# compute cumulative totals
denv_100k = denv_100k.groupby(by=f'{region}')['DENV_per_100k'].sum()
# standardize
sc = StandardScaler()
geography['denv_100k_cumulative'] = sc.fit_transform(denv_100k.values.reshape(-1,1))
# add to covariate name mapping
if include_denv_100k_cumulative:
    covariate_names.extend(['denv_100k_cumulative',])



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

# Add to covariate name mapping
if include_compactness:
    covariate_names.extend(['cx', 'cy'])



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
# Add to covariate name mapping
if include_denv_100k_DTW:
    covariate_names.extend(DTW_covariates_denv_100k_names)



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
# Add to covariate name mapping
if include_indexP_DTW:
    covariate_names.extend(DTW_covariates_indexP_names)



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
# Add to covariate name mapping
if include_serotypes_DTW:
    covariate_names.extend(DTW_covariates_serotypes_names)



# Run max-p regionalization model `n` times in parallel
# TODO: when packaging this code these 2 functions will go in the source code
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def run_single_maxp(
    run_index, geography_df, region_col, covariate_names, threshold,
    max_iterations_sa
):
    """
    Worker function: rebuilds its own MaxP model & captures stdout inside process.
    Returns (run_index, labels, co_assoc_matrix, best_obj_value)
    """
    # Local import to avoid multiprocessing overhead
    from libpysal.weights import Rook
    from spopt.region import MaxPHeuristic

    # Build contiguity weights
    w = Rook.from_dataframe(geography_df, use_index=False)

    # Build model
    model = MaxPHeuristic(
        geography_df,
        w,
        attrs_name=covariate_names,
        threshold_name="N_typed_yearly_median",
        threshold=threshold,
        top_n=2,
        policy="multiple",
        max_iterations_construction=1000,
        max_iterations_sa=max_iterations_sa,
        verbose=True
    )

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        model.solve()

    stdout_data = f.getvalue().splitlines()
    best_vals = []

    # Parse stdout for objective value
    for i, line in enumerate(stdout_data):
        if "best objective value:" in line.lower():
            try:
                best_vals.append(float(stdout_data[i+1].strip()))
            except:
                pass
    best_obj_value = best_vals[-1] if best_vals else np.nan

    # Build outputs
    labels = model.labels_.copy()
    coassoc = build_co_association_matrix(geography_df[region_col], labels)

    return run_index, labels, coassoc, best_obj_value


def run_parallel_maxp(n_cores, n, geography, region, covariate_names, threshold, max_iterations_sa):

    results = []
    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        futures = [
            ex.submit(
                run_single_maxp,
                run_index=i,
                geography_df=geography,
                region_col=region,
                covariate_names=covariate_names,
                threshold=threshold,
                max_iterations_sa=max_iterations_sa,
            )
            for i in range(n)
        ]

        for fut in futures:
            results.append(fut.result())

    # Sort by run index so output ordering is guaranteed
    results.sort(key=lambda x: x[0])

    # Extract outputs
    all_labels   = [r[1] for r in results]
    all_matrices = [r[2] for r in results]
    obj_vals     = [r[3] for r in results]

    return all_labels, all_matrices, obj_vals

# run model in parallel
if __name__ == '__main__':
    # default unix parallel process spawn method
    mp.set_start_method("fork")
    # run func
    labels, matrices, best_obj_vals = run_parallel_maxp(
        n_cores=mp.cpu_count(),
        n=n,
        max_iterations_sa=max_iterations_sa,
        geography=geography,
        region=f"{region}",
        covariate_names=covariate_names,
        threshold=threshold,
    )



# Assign weights to every run using tuned softmax
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

T = (50/1300) * (np.mean(best_obj_vals)) # Trying 50/1300
weights = softmax(-np.asarray(best_obj_vals)/T)



# Save individual runs in geography dataframe
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

for i in range(n):
    geography[f'run_{i}'] = labels[i]



# Average co-association matrices across runs
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# compute softmax-weighted mean co-association matrix
prob_matrix = pd.DataFrame(0.0, index=geography[region], columns=geography[region])
for association_matrix, weight in zip(matrices, weights):
   prob_matrix += weight * association_matrix

# save mean co-association matrix
prob_matrix.to_csv(os.path.join(output_folder, f"prob_matrix_{spatial_aggregation}.csv"))

# compute median number of clusters
n_clusters = int(np.median([len(np.unique(l)) for l in labels]))

# make a categorical color palette with n_clusters distinct colors
glasbey_cmap = ListedColormap(create_palette(palette_size=n_clusters))



# Randomly select and visualize 12 runs on a 3x4 grid
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# 1. Get all run columns
run_columns = [col for col in geography.columns if col.startswith("run_")]
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
plt.savefig(os.path.join(output_folder, f'clusters_{spatial_aggregation}.png'), dpi=300)
plt.close()



# Recluster mean co-association matrix using hierarchical clustering
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# hierarchical clustering needs distance matrix
distance = 1 - prob_matrix
distance = np.clip(distance, 0, 1) # some distances may become very very small negative numbers 

# Perform hierarchical clustering (average linkage)
Z = linkage(squareform(distance, checks=False), method='average')

# Choose number of clusters k
geography['consensus_clusters_hierarchical'] = fcluster(Z, n_clusters, criterion='maxclust')

# Save clustermap
import seaborn as sns
sns.clustermap(1-distance, cmap='viridis')
plt.savefig(os.path.join(output_folder, f'clustermap_probmatrix_{spatial_aggregation}.png'), dpi=300)
plt.close()



# Recluster mean co-association matrix using spectral clustering
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
plt.savefig(os.path.join(output_folder, f'consensus_clusters_{spatial_aggregation}.png'), dpi=300)
plt.close()

# Save the consensus clusters (hierarchical)
clusters = geography[[f'{region}', 'consensus_clusters_hierarchical']]
clusters = clusters.rename(columns={'consensus_clusters_hierarchical': 'cluster'})
clusters.to_csv(os.path.join(output_folder, f"clusters_{spatial_aggregation}.csv"), index=False)



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
adj_matrix.to_csv(os.path.join(output_folder, f'adjacency_matrix_{spatial_aggregation}.csv'))



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
dist_matrix.to_csv(os.path.join(output_folder, f'distance_matrix_{spatial_aggregation}.csv'))