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
year = 2010
if year:
    demographics = demographics[demographics['year']==year][['geocode','population']]
else:
    demographics = demographics.groupby(by='geocode')['population'].mean()

# Merge demography with the geodata
gdf = gdf.merge(demographics, on="geocode")


# Build a state adjacency matrix
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Step 1: Clean invalid geometries
gdf = gdf[gdf.geometry.notnull()].copy()
gdf["geometry"] = gdf["geometry"].buffer(0)

# Step 2: Dissolve municipalities to state-level geometries
states_gdf = gdf.dissolve(by="uf", as_index=False)
states_gdf = states_gdf.reset_index(drop=True)

# Step 3: Ensure "uf" column is clean and sorted
states_gdf["uf"] = states_gdf["uf"].astype(str).str.strip()
states_gdf = states_gdf.sort_values("uf").reset_index(drop=True)
uf_list = states_gdf["uf"].tolist()

# Step 4: Build spatial index and adjacency dictionary
sindex = states_gdf.sindex
adjacency = {uf: set() for uf in uf_list}

for i, row in states_gdf.iterrows():
    geom_i = row.geometry
    uf_i = row["uf"]
    possible_matches_index = list(sindex.intersection(geom_i.bounds))
    
    for j in possible_matches_index:
        if i == j:
            continue
        geom_j = states_gdf.loc[j, "geometry"]
        uf_j = states_gdf.loc[j, "uf"]
        
        # Use intersects instead of touches for robustness
        if geom_i.intersects(geom_j):
            adjacency[uf_i].add(uf_j)
            adjacency[uf_j].add(uf_i)  # symmetric

# Step 5: Convert to binary adjacency matrix
adj_matrix = pd.DataFrame(0, index=uf_list, columns=uf_list)

for uf in uf_list:
    for neighbor in adjacency[uf]:
        adj_matrix.loc[uf, neighbor] = 1

# Save in a .csv
adj_matrix.to_csv('../interim/adjacency_matrix.csv')


# Build a demographically weighted distance matrix
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Assure appropriate projection
gdf = gdf.to_crs("EPSG:5880")

# Calculate centroids of municipalities
gdf['centroid'] = gdf.geometry.centroid

# Helper function
def weighted_centroid(group):
    # Get x and y from centroid
    x = group['centroid'].x
    y = group['centroid'].y
    weights = group['population']
    
    # Weighted average
    x_bar = (x * weights).sum() / weights.sum()
    y_bar = (y * weights).sum() / weights.sum()
    
    return Point(x_bar, y_bar)

# Group by UF and calculate weighted centroids
weighted_centroids = gdf.groupby('uf').apply(weighted_centroid).reset_index()
weighted_centroids.columns = ['uf', 'geometry']
uf_centroids_gdf = gpd.GeoDataFrame(weighted_centroids, geometry='geometry', crs=gdf.crs)

# Create empty DataFrame
uf_codes = uf_centroids_gdf['uf'].tolist()
dist_matrix = pd.DataFrame(index=uf_codes, columns=uf_codes, dtype=float)

# Fill with distances in kilometers
for i, row_i in uf_centroids_gdf.iterrows():
    for j, row_j in uf_centroids_gdf.iterrows():
        dist = row_i.geometry.distance(row_j.geometry) / 1000  # meters to km
        dist_matrix.loc[row_i['uf'], row_j['uf']] = dist

# Save the distance matrix to a csv file
dist_matrix.to_csv('../interim/weighted_distance_matrix.csv')