import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from spopt.region import MaxPHeuristic
from libpysal.weights import Rook, Queen
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

# spatial aggregation: 'mun' (5570 municipalities), 'rgi' (508 immediate regions), 'rgint' (130 intermediate regions)
region_filename = 'rgint'

# Load raw data
# >>>>>>>>>>>>>

# Load geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")

# Load case data
denv = pd.read_csv('../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv')
denv['date'] = pd.to_datetime(denv['date'])

# Load DTW-MDS embedding
DTW_covariates = pd.read_csv(f'../../data/interim/DTW-MDS-embeddings/DTW-MDS-embedding_{region_filename}.csv')
region = DTW_covariates.columns.to_list()[0]


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
    # --- 2. Dissolve geometries by immediate region ---
    gdf_regions = geography.dissolve(by=f'{region}', aggfunc={'POP': 'sum'})
    # --- 3. Attach the majority biome back ---
    gdf_regions['biome'] = gdf_regions.index.map(biome_majority)
    # --- 4. Retain only relevant columns ---
    gdf_regions = gdf_regions.reset_index()
    geography = gdf_regions[[f'{region}', 'biome', 'POP', 'geometry']]

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
# limit time window (from 2020 onwards all regions have good subtyping)
denv = denv[((denv['date'] > datetime(1900,1,1)) & (denv['date'] < datetime(2020,1,1)))]
# extract year
denv["year"] = pd.to_datetime(denv["date"]).dt.year
# compute total cases per month
denv["N_typed"] = denv[["DENV_1","DENV_2","DENV_3","DENV_4"]].sum(axis=1)
# sum cases by year
active_sum = denv.groupby([f'{region}',"year"])['N_typed'].sum().reset_index()
# take mean across years
mean_active_sum = active_sum.groupby(f'{region}')["N_typed"].mean().reset_index()
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



# Make DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates, 
    on = f'{region}'
)

# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates = [x for x in DTW_covariates.columns.to_list() if x != f'{region}']
geography[DTW_covariates] = sc.fit_transform(geography[DTW_covariates])



# Decide on attributes to use
# >>>>>>>>>>>>>>>>>>>>>>>>>>>

# my pick
attrs = DTW_covariates + ['cx', 'cy'] #+ [region+'_NORM'] #+ biome_dummies.columns.to_list() 



# Build weights matrix
# >>>>>>>>>>>>>>>>>>>>>

# Build contiguity weight map
w = Rook.from_dataframe(geography)



# Setup and run the max-p model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

threshold = 50  # Sum of column 'N_typed_monthly_mean' should exceed this threshold in every cluster
model = MaxPHeuristic(
    geography,
    w, 
    attrs_name=attrs,
    threshold_name='N_typed_monthly_mean',
    threshold=threshold,
    top_n=3,
    verbose=True,
    policy='multiple',
    max_iterations_construction=1000,
    max_iterations_sa=100,
)
model.solve()



# Save and visualise the clustering results
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# add cluster labels to GeoDataFrame
geography["cluster"] = model.labels_

# visualise clusters on a map
fig, ax = plt.subplots(figsize=(10, 10))
geography.plot(
    column="cluster",          # color regions by cluster label
    categorical=True,
    cmap="tab20",             # categorical colormap
    linewidth=0.1,
    edgecolor="grey",
    legend=True,
    ax=ax
)
ax.set_title("Max-p Regionalization of Brazilian Municipalities", fontsize=14)
ax.axis("off")
plt.savefig(f'../../data/interim/clusters/clusters_{region_filename}.png', dpi=200)
plt.show()
plt.close()

# save the result
geography[[f'{region}', 'cluster']].to_csv(f'../../data/interim/clusters/clusters_{region_filename}.csv', index=False)



# Build the clusters adjacency matrix needed for the Bayesian imputation model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Step 1: Dissolve municipalities to state-level geometries
clusters_geography = geography.dissolve(by='cluster', as_index=False)
clusters_geography = clusters_geography.reset_index(drop=True)

# Step 2: Ensure 'cluster' column is sorted
clusters_geography = clusters_geography.sort_values('cluster').reset_index(drop=True)
cluster_list = clusters_geography['cluster'].tolist()

# Step 4: Build spatial index and adjacency dictionary
sindex = clusters_geography.sindex
adjacency = {idx: set() for idx in cluster_list}

for i, row in clusters_geography.iterrows():
    geom_i = row.geometry
    uf_i = row['cluster']
    possible_matches_index = list(sindex.intersection(geom_i.bounds))
    
    for j in possible_matches_index:
        if i == j:
            continue
        geom_j = clusters_geography.loc[j, "geometry"]
        uf_j = clusters_geography.loc[j, 'cluster']
        
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
