import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
import matplotlib.pyplot as plt
from datetime import datetime


# Fetch the municipality shapefiles and demographics and merge them
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Load geodata
gdf = gpd.read_file("../raw/sprint_2025/shape_muni.gpkg", layer="shape_muni")

# Load demography
demographics = pd.read_csv('../raw/sprint_2025/datasus_population_2001_2024.csv')

# Choose year or average
year = None
if year:
    demographics = demographics[demographics['year']==year][['geocode','population']]
else:
    demographics = demographics.groupby(by='geocode')['population'].mean()

# Merge demography with the geodata
gdf = gdf.merge(demographics, on="geocode")

# rename to pop
gdf = gdf.rename(columns={'population': 'pop'})


# Compute population density
# >>>>>>>>>>>>>>>>>>>>>>>>>>

# Reproject to equal-area CRS for Brazil
gdf_eq = gdf.to_crs(epsg=5880)

# Compute area in km²
gdf_eq["area_km2"] = gdf_eq.geometry.area / 1e6

# Population density (people per km²)
gdf["pop_density"] = gdf_eq["pop"] / gdf_eq["area_km2"]


# Attach Koppen classification
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>

# load data
koppen = pd.read_csv('../raw/sprint_2025/environ_vars.csv')[['geocode', 'koppen', 'biome']]

# merge with the geodata
gdf = gdf.merge(koppen, on="geocode")


# Attach ...
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>

# insert any other clustering covariates here


# Save result
# >>>>>>>>>>>

# Save to a GeoPackage
gdf.to_file("../interim/geographic-dataset.gpkg", driver="GPKG")