import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from spopt.region import MaxPHeuristic
from libpysal.weights import Rook, Queen
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler


region = 'immediate'

# Load raw data
# >>>>>>>>>>>>>

# Load geodata
geography = gpd.read_file("../../data/interim/geographic-dataset.gpkg")

# Load case data
denv = pd.read_csv('../../data/interim/datasus_DENV-linelist/municipality/DENV-serotypes_1996-2025_monthly_municipality.csv', parse_dates=True)


# Aggregate to the intermediate/immediate regions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if region:

    assert ((region != 'immediate') | (region != 'intermediate')), "'region' must be either 'immediate' or 'intermediate''"

    # Geography
    # >>>>>>>>>

    muncipality_region_map = geography[['geocode', f'{region}_region_name']]

    # --- 1. Majority vote of biome per immediate region ---
    # Count how many municipalities per biome in each immediate region
    biome_majority = (
        geography.groupby([f'{region}_region_name', 'biome'])
        .size()
        .reset_index(name='count')
    )
    # For each immediate region, keep the biome with max count
    biome_majority = (
        biome_majority
        .sort_values([f'{region}_region_name', 'count'], ascending=[True, False])
        .drop_duplicates(f'{region}_region_name')
        .set_index(f'{region}_region_name')['biome']
    )
    # --- 2. Dissolve geometries by immediate region ---
    gdf_regions = geography.dissolve(by=f'{region}_region_name', aggfunc={'pop': 'sum'})
    # --- 3. Attach the majority biome back ---
    gdf_regions['biome'] = gdf_regions.index.map(biome_majority)
    # --- 4. Retain only relevant columns ---
    gdf_regions = gdf_regions.reset_index()
    geography = gdf_regions[[f'{region}_region_name', 'biome', 'pop', 'geometry']]
    # --- 5. Rename 'immediate_region_name' to 'geocode' ---
    geography = geography.rename(columns={f'{region}_region_name': 'geocode'})

    # Incidence
    # >>>>>>>>>

    # Merge incidence with mapping
    denv = denv.merge(muncipality_region_map, on="geocode", how="left")
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
        denv.groupby([f"{region}_region_name", "date"])[denv_cols]
        .agg(nan_to_zero_sum)
        .reset_index()
    )
    denv = denv.rename(columns={f'{region}_region_name': 'geocode'})
    # Start in 1999
    denv['date'] = pd.to_datetime(denv['date'])
    denv = denv[denv['date'] > datetime(2000,1,1)]


# Dynamic Time Warping
# >>>>>>>>>>>>>>>>>>>>

# normalize total dengue cases to incidence per 100K
total_DENV = denv[['geocode', 'date', 'DENV_total']]
total_DENV = total_DENV.merge(geography[['geocode', 'pop']], on="geocode", how="left")
total_DENV["DENV_per_100k"] = (
    total_DENV["DENV_total"] / total_DENV["pop"] * 1e5
)


# Compute threshold
# >>>>>>>>>>>>>>>>>

# Compute the mimimum sum of serotyped cases across all years (will have to be changed)
# Extract year
denv["year"] = pd.to_datetime(denv["date"]).dt.year
# Compute total cases per month
denv["total_cases"] = denv[["DENV_1","DENV_2","DENV_3","DENV_4"]].sum(axis=1)

# Sum cases during active months by year
active_sum = denv.groupby(["geocode","year"]).apply(lambda x: x.loc[x.total_cases>0,"total_cases"].sum()).reset_index(name="active_sum")

# Take mean across years
mean_active_sum = active_sum.groupby("geocode")["active_sum"].mean().reset_index()
mean_active_sum.rename(columns={"active_sum":"mean_active_sum"}, inplace=True)

# Merge min_yearly_sum
geography = geography.merge(mean_active_sum, on="geocode", how="left")


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

# 4) Choose alpha (compactness weight). Try small values first.
alpha = 1   # try different values
geography["cx"] *= alpha 
geography["cy"] *= alpha

# 5) Add these to your attrs list for MaxPHeuristic
attrs = biome_dummies.columns.tolist() + ["cx", "cy"]

# Build weights matrix
# >>>>>>>>>>>>>>>>>>>>>

# Build Queen contiguity
w = Rook.from_dataframe(geography)


# Setup and run the max-p model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

threshold = 20  # Your chosen minimum sum per cluster
model = MaxPHeuristic(
    geography,
    w, 
    attrs_name=attrs,
    threshold_name='mean_active_sum',
    threshold=threshold,
    top_n=2,
    verbose=True,
    policy='multiple',
    max_iterations_construction=10000,
    max_iterations_sa=50,
)
model.solve()


# Save and visualise the results
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Add cluster labels to GeoDataFrame
geography["clusters"] = model.labels_

fig, ax = plt.subplots(figsize=(10, 10))
geography.plot(
    column="clusters",          # color regions by cluster label
    categorical=True,
    cmap="tab20",             # categorical colormap
    linewidth=0.1,
    edgecolor="grey",
    legend=True,
    ax=ax
)
ax.set_title("Max-p Regionalization of Brazilian Municipalities", fontsize=14)
ax.axis("off")
plt.show()
plt.close()

geography[['geocode', 'clusters']].to_csv('../../data/interim/clusters.csv')