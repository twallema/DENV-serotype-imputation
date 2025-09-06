import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
import matplotlib.pyplot as plt
from datetime import datetime


# Fetch the municipality shapefiles and demographics and merge them
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Load geodata by IBGE
gdf = gpd.read_file("../raw/BR_Municipios_2023/BR_Municipios_2023.shp")[['CD_MUN', 'NM_MUN', 'CD_RGI', 'NM_RGI', 'CD_RGINT', 'NM_RGINT', 'CD_UF', 'NM_UF', 'CD_REGIAO', 'NM_REGIAO', 'geometry']]

# Convert unit codes to numeric type
for col in ['CD_MUN', 'CD_RGI', 'CD_RGINT', 'CD_UF', 'CD_REGIAO']:
    gdf[col] = pd.to_numeric(gdf[col])

# Load demography
demographics = pd.read_csv('../raw/sprint_2025/datasus_population_2001_2024.csv')
demographics = demographics.rename(columns={'geocode': 'CD_MUN'})

# Choose year or average
year = None
if year:
    demographics = demographics[demographics['year']==year][['CD_MUN','population']]
else:
    demographics = demographics.groupby(by='CD_MUN')['population'].mean()

# Merge demography with the geodata
gdf = gdf.merge(demographics, on="CD_MUN")

# rename to pop
gdf = gdf.rename(columns={'population': 'POP'})

# save a lightweight csv copy of the mapping between codes
pd.DataFrame(gdf[['CD_MUN', 'NM_MUN', 'CD_RGI', 'NM_RGI', 'CD_RGINT', 'NM_RGINT', 'CD_UF', 'NM_UF', 'CD_REGIAO', 'NM_REGIAO']]).to_csv('../interim/spatial_units_mapping.csv', index=False)

# Compute population density
# >>>>>>>>>>>>>>>>>>>>>>>>>>

# Reproject to equal-area CRS for Brazil
gdf_eq = gdf.to_crs(epsg=5880)

# Compute area in km²
gdf_eq["area_km2"] = gdf_eq.geometry.area / 1e6

# Population density (people per km²)
gdf["POP_DENS"] = gdf_eq["POP"] / gdf_eq["area_km2"]

# Attach Koppen classification and Biome type
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# load data
environ_vars = pd.read_csv('../raw/sprint_2025/environ_vars.csv')[['geocode', 'koppen', 'biome']]
environ_vars = environ_vars.rename(columns={'geocode': 'CD_MUN'})

# merge with the geodata
gdf = gdf.merge(environ_vars, on="CD_MUN")

# Attach ...
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>

# insert any other clustering covariates here


# Save result
# >>>>>>>>>>>

# To avoid geometry dissolution issues later down the line
gdf["geometry"] = gdf.geometry.apply(shapely.make_valid)

# Convert codes to int
gdf['CD_RGI'] = gdf['CD_RGI'].astype(int)
gdf['CD_RGINT'] = gdf['CD_RGINT'].astype(int)
gdf['CD_UF'] = gdf['CD_UF'].astype(int)
gdf['CD_REGIAO'] = gdf['CD_REGIAO'].astype(int)

# Save to a GeoPackage
gdf.to_parquet("../interim/geographic-dataset.parquet", compression='brotli')