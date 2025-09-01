import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from spopt.region import MaxPHeuristic
from libpysal.weights import Rook, Queen
from sklearn.preprocessing import StandardScaler


# Load data
# >>>>>>>>>

# Load geodata
geography = gpd.read_file("../../data/interim/geographic-dataset.gpkg")



# Load case data
denv = pd.read_csv('../../data/interim/datasus_DENV-linelist/municipality/DENV-serotypes_1996-2025_monthly_municipality.csv')


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
    top_n=1,
    verbose=True,
    policy='multiple',
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