import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
import matplotlib.pyplot as plt
from datetime import datetime


# Fetch the municipality shapefiles and demographics and merge them
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Load geodata by sprint 2025
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


# Attach the immediate and intermediate regions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Load geodata by sprint 2025
regions = gpd.read_file("../raw/RG2017_regioesgeograficas2017_20180911/RG2017_regioesgeograficas2017.shp")

# Rename columns to facilitate merging
regions = regions.rename(columns={'CD_GEOCODI': 'geocode', 'nome_rgi': 'immediate_region_name', 'nome_rgint': 'intermediate_region_name'})

# Set geocde to int
regions['geocode'] = regions['geocode'].astype(int)

# Merge demography with the geodata
gdf = gdf.merge(regions[['geocode', 'immediate_region_name', 'intermediate_region_name']], on="geocode")


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

# To avoid geometry dissolution issues later down the line
gdf["geometry"] = gdf.geometry.apply(shapely.make_valid)

# Save to a GeoPackage
gdf.to_file("../interim/geographic-dataset.gpkg", driver="GPKG")