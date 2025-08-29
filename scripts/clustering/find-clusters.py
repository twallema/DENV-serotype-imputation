import spopt
import libpysal
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load geodata
geography = gpd.read_file("../../data/interim/geographic-dataset.gpkg")

# Load case data
denv = pd.read_csv('../../data/interim/datasus_DENV-linelist/municipality/DENV-serotypes_1996-2025_monthly_municipality.csv')



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

# Make dummies for the biome
biome_dummies = pd.get_dummies(geography["biome"], prefix="biome")
geography = geography.merge(
    biome_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)

from libpysal.weights import Queen
# Project to Brazil Polyconic
geography = geography.to_crs(epsg=5880)
# Build Queen contiguity
w = Queen.from_dataframe(geography)


# Set up model
from spopt.region import MaxPHeuristic
threshold = 20  # Your chosen minimum sum per cluster
model = MaxPHeuristic(
    geography,
    w, 
    attrs_name=biome_dummies.columns.tolist(),
    threshold_name='mean_active_sum',
    threshold=threshold,
    top_n=1,
    verbose=True
)
model.solve()


# Add cluster labels to GeoDataFrame
geography["region"] = model.labels_



fig, ax = plt.subplots(figsize=(10, 10))
geography.plot(
    column="region",          # color regions by cluster label
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

