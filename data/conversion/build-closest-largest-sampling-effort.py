import os
import polars as pl
import numpy as np
import pandas as pd
import geopandas as gpd
from glasbey import create_palette
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


##############
## Settings ##
##############

# spatial aggregation
region_filename = 'rgint'  
region = 'CD_RGINT'         

start_month_season = 9


##############################################################
## Compute median serotyping effort per region over seasons ##
##############################################################

# get mapping
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
geography = geography.to_crs("EPSG:5880")

muncipality_region_map = geography[['CD_MUN', f'{region}']].set_index('CD_MUN').to_dict()[f'{region}']

# write a NaN-retaining aggregation function
agg_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]
agg_exprs = []
for c in agg_cols: 
    agg_exprs.extend([
        pl.col(c).sum().alias(c),
        pl.col(c).count().alias(f"{c}_count"),  
    ])

# get case data
cases = (
    pl.scan_parquet("../../data/interim/datasus_DENV-linelist/DENV-1999_2026-month-mun-no_diagnostics.parquet")
    # aggregate to regions
    .with_columns(pl.col("CD_MUN").replace_strict(muncipality_region_map).alias(f"{region}"))
    .group_by(["date", f"{region}"])
    .agg(agg_exprs)
    .with_columns([
        pl.when(pl.col(f"{c}_count") == 0)
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in agg_cols
    ])
    # count serotyping effort
    .with_columns(
        DENV_serotyped_count=(
            pl.when(pl.sum_horizontal("^DENV_[1-4]$") == 0)
            .then(None)
            .otherwise(pl.sum_horizontal("^DENV_[1-4]$"))
        )
    )
    .drop([f"{c}_count" for c in agg_cols])
    .sort(["date", f"{region}"])
    .collect(engine="streaming")
)

# append a "season" label (September of year X -> September of year X+1)
cases_season = (
    cases
    .with_columns(
        # Determine the starting year of the season
        pl.when(pl.col("date").dt.month() >= start_month_season)
        .then(pl.col("date").dt.year())
        .otherwise(pl.col("date").dt.year() - 1)
        .alias("season_start")
    )
    .with_columns(
        # Format as "YYYY-YYYY" string label
        pl.concat_str([pl.col("season_start"), pl.col("season_start") + 1], separator="-").alias("season")
    )
    .drop("season_start")
    .filter(pl.col("season") != "2026-2027")
)

## compute serotyped cases per season
cases_season = (
    cases_season.group_by(["season", f"{region}"])
    .agg(pl.sum_horizontal("^DENV_[1-4]$").alias('total_serotyped').sum().alias('total_serotyped'))
    .sort(["season", f"{region}"])
)

## compute summary statistics (median, IQR, skew) across seasons per cluster
cases_season_stats = (
    cases_season
    .group_by(f"{region}")
    .agg([
        pl.col("total_serotyped").median().alias("median"),
        pl.col("total_serotyped").quantile(0.75).alias("q3"),
        pl.col("total_serotyped").quantile(0.25).alias("q1"),
        pl.col("total_serotyped").skew().alias("skew"),
    ])
    .with_columns([
        (pl.col("q3") - pl.col("q1")).alias("iqr"),
        # Quartile Coefficient of Dispersion (Relative IQR)
        pl.when((pl.col("q3") + pl.col("q1")) == 0)
        .then(1.0)
        .otherwise((pl.col("q3") - pl.col("q1")) / (pl.col("q3") + pl.col("q1")))
        .alias("qcd")
    ])
    .select([f'{region}', 'median', 'skew', 'qcd'])
)

########################################
## Perform selection based on quality ##
########################################

good_regions_quality = pd.read_csv("../../data/interim/DTW-MDS-embeddings/serotypes/serotype_trajectory_quality.csv")


#####################################################
## Perform selection based on quantitative metrics ##
#####################################################

# ranking = (
#     cases_season_stats
#     .with_columns([
#         # Percentile ranks (0.0 to 1.0)
#         (pl.col("median").rank("dense") / pl.count()).alias("rank_median"),
#         (pl.col("qcd").rank("dense", descending=True) / pl.count()).alias("rank_qcd"), # lower QCD is better
#         (pl.col("skew").fill_null(99).rank("dense", descending=True) / pl.count()).alias("rank_skew"), # lower skew is better
#     ])
#     .with_columns(
#         # Weighted Composite Score (adjust weights if median matters more than skew)
#         serotyping_index = (
#             1.0 * pl.col("rank_median") + 
#             0.0 * pl.col("rank_qcd") + 
#             0.0 * pl.col("rank_skew")
#         )
#     )
#     .sort("serotyping_index", descending=True)
# )

# good_regions_filtered = ranking[:len(good_regions_quality[good_regions_quality['has_quality'] == 1])]


####################
## Visualise them ##
####################

# geography_states = geography.dissolve(by='CD_UF')

# geography_regions = geography.dissolve(by=f'{region}').reset_index()

# geography_regions.loc[geography_regions[f"{region}"].isin(good_regions_filtered.select(pl.col(f"{region}")).to_series().to_list()), 'cluster'] = 'quantitative'
# geography_regions.loc[geography_regions[f"{region}"].isin(good_regions_quality.loc[good_regions_quality['has_quality']==1, f"{region}"]), 'cluster'] = 'qualitative'
# geography_regions.loc[((geography_regions[f"{region}"].isin(good_regions_filtered.select(pl.col(f"{region}")).to_series().to_list())) & (geography_regions[f"{region}"].isin(good_regions_quality.loc[good_regions_quality['has_quality']==1, f"{region}"]))), 'cluster'] =  'overlap'
# geography_regions['cluster'] = geography_regions['cluster'].fillna('other')

# fig, ax = plt.subplots()
# geography_states.boundary.plot(ax=ax, linewidth=0.5, color="black")   # state boundaries
# geography_regions.plot(
#     column="cluster",          # color regions by cluster label
#     categorical=True,
#     linewidth=0.2,
#     edgecolor="grey",
#     legend=True,
#     ax=ax,
#     legend_kwds={'fontsize': 4, 'ncol': 2, 'loc': 'lower right', 'markerscale': 0.4}
# )
# ax.set_title(f"Median number of serotyped cases per season", fontsize=12)
# ax.axis("off")
# os.makedirs(f'../../data/interim/nearest-largest-sampling-effort/', exist_ok=True)
# plt.savefig(f'../../data/interim/nearest-largest-sampling-effort/qualitative-quantitative-comparison_{region_filename}.png', dpi=600)
# plt.close()

#####################
## Turn into a map ##
#####################

# load the DTW distance
dtw_matrix = pd.read_csv(f'../../data/interim/DTW-MDS-embeddings/serotypes/dtw/dtw-matrix_post-2020_{region}.csv', index_col=0).values

# make a distance matrix of the regions
geography_regions = geography.dissolve(by=f'{region}').reset_index()
from scipy.spatial.distance import cdist
## get the X and Y coordinates of each polygon's centroid
centroids = geography_regions.geometry.centroid
points = list(zip(centroids.x, centroids.y))
## compute the square distance matrix
dist_array = cdist(points, points, metric='euclidean')/1000
## convert to a readable DataFrame with matching IDs
distance_matrix = pd.DataFrame(
    dist_array, 
    index=geography_regions['CD_RGINT'], 
    columns=geography_regions['CD_RGINT']
).values


geography_regions_copy = geography_regions.copy(deep=True)

gammas = [0, 0.005, 0.1]

for gamma in gammas:

    gdf = geography_regions_copy

    # convolve DTW distance and geographic distance
    weighted_dtw = pd.DataFrame(
        data = dtw_matrix * np.exp(-gamma * distance_matrix),
        index = geography_regions[f'{region}'], 
        columns = geography_regions[f'{region}']
    )

    # for each region select the closest match
    good_regions = good_regions_quality[good_regions_quality["has_quality"] == 1][f"{region}"]
    gdf['closest'] = np.nan

    for idx, region_id in enumerate(gdf[f'{region}']):

        if region_id not in good_regions.values:
            geography_regions.loc[geography_regions[f'{region}'] == region_id, 'closest'] = int(weighted_dtw.loc[region_id, good_regions.values].idxmax())
        else:
            geography_regions.loc[geography_regions[f'{region}'] == region_id, 'closest'] = region_id

    # merge the good quality regions
    gdf = geography_regions.merge(good_regions_quality, on=f"{region}")

    import networkx as nx

    # Define your region ID column variable for flexibility
    region_col = f"{region}"  # Update this to match your variable if named differently

    # Step 1: Identify all regions that have quality == 1
    quality_gdf = gdf[gdf['has_quality'] == 1].copy()

    # Step 2: Group adjacent "has_quality" regions into unified cluster components using spatial touch/overlaps
    # We create a temporary spatial adjacency graph for quality regions
    quality_gdf['temp_geom_idx'] = quality_gdf.index
    spatial_adj_graph = nx.Graph()

    for idx, row in quality_gdf.iterrows():
        spatial_adj_graph.add_node(row[region_col])

    # Find which quality regions touch each other
    for i, row1 in quality_gdf.iterrows():
        for j, row2 in quality_gdf.iterrows():
            if i < j and row1['geometry'].touches(row2['geometry']):
                spatial_adj_graph.add_edge(row1[region_col], row2[region_col])

    # Find connected components of adjacent quality regions (these form unified cluster centers)
    quality_components = list(nx.connected_components(spatial_adj_graph))

    # Map every quality region to a representative root ID (e.g., the first item or a merged string/id)
    quality_to_center = {}
    for component in quality_components:
        # Choose a deterministic representative center ID for this group of adjacent quality regions
        center_id = sorted(list(component))[0] 
        for region_id in component:
            quality_to_center[region_id] = center_id

    # Flatten list of all valid cluster center root IDs
    cluster_centers = list(set(quality_to_center.values()))

    # Step 3: Build the Directed Graph for the rest of the matching logic
    G = nx.DiGraph()

    for idx, row in gdf.iterrows():
        G.add_node(row[region_col], has_quality=row['has_quality'], geometry=row['geometry'])

    for idx, row in gdf.iterrows():
        region_id = row[region_col]
        closest_target = row['closest']
        
        # If the region is a quality region, point it directly to its group's unified center root
        if row['has_quality'] == 1:
            target = quality_to_center[region_id]
            if region_id != target:
                G.add_edge(region_id, target)
        elif pd.notna(closest_target) and region_id != closest_target:
            G.add_edge(region_id, closest_target)

    # Step 4: Assign each region to its ultimate cluster center root
    region_to_cluster = {}

    for node in G.nodes():
        current = node
        visited = set()
        
        while current not in visited:
            visited.add(current)
            
            # If we hit a unified cluster center root
            if current in cluster_centers:
                region_to_cluster[node] = current
                break
                
            neighbors = list(G.successors(current))
            if not neighbors:
                region_to_cluster[node] = current 
                break
            current = neighbors[0]
        else:
            region_to_cluster[node] = node

    # Step 5: Map the cluster assignments back to your main GeoDataFrame
    geography_regions[f'cluster_id_{gamma}'] = gdf[region_col].map(region_to_cluster)
    geography_regions[f"cluster_id_{gamma}"] = pd.factorize(geography_regions[f"cluster_id_{gamma}"])[0]


# Visualise on a map 
glasbey_cmap = ListedColormap(create_palette(palette_size=len(good_regions_quality[good_regions_quality['has_quality'] == 1])))

geography_states = geography.dissolve(by='CD_UF')
geography_regions = geography_regions.merge(good_regions_quality, on="CD_RGINT")

titles = ["DTW distance", f"Blend ($\\gamma$ = {gammas[1]})", "Geographic distance"]

fig, ax = plt.subplots(ncols=3, figsize=(11.7, 8.3/2))

for i, gamma in enumerate(gammas):

    geography_states.boundary.plot(ax=ax[i], linewidth=0.5, color="black")   # state boundaries

    geography_regions.plot(
        column=f"cluster_id_{gamma}",          # color regions by cluster label
        categorical=True,
        cmap = glasbey_cmap,
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        ax=ax[i],
        legend_kwds={'fontsize': 4, 'ncol': 2, 'loc': 'lower right', 'markerscale': 0.4}
    )

    quality_regions = geography_regions[geography_regions['has_quality'] == 1]
    quality_regions.plot(
        ax=ax[i],
        color="grey",
        edgecolor="black",
        hatch="////",
        linewidth=0.5,
        alpha=0.4  # Slight transparency softens the grey
    )

    ax[i].set_title(titles[i])

os.makedirs(f'../../data/interim/nearest-largest-sampling-effort/', exist_ok=True)
plt.savefig(f'../../data/interim/nearest-largest-sampling-effort/nearest-largest-sampling-effort_{region_filename}.png', dpi=600)
plt.close()

#################
## Save output ##
#################

gamma_save = 0.1

output = geography_regions[[f'{region}', f'cluster_id_{gamma_save}']]
output = output.rename(columns={f'cluster_id_{gamma_save}': 'largest_sampling_effort_id'})
output.to_csv(f'../../data/interim/nearest-largest-sampling-effort/nearest-largest-sampling-effort_{region_filename}.csv', index=False)


#######################################################################################
## Visualise data in these clusters with and without removal of low-data states data ##
#######################################################################################

# make a mapping from f'{region}' to the newly defined closest-largest-sampling-effort regions
region_cluster_map = output.set_index(f'{region}').to_dict()['largest_sampling_effort_id']

# get regions with insufficient data
threshold = 2
insufficient_regions = cases_season_stats.filter(pl.col("median") <= threshold).select(pl.col(f"{region}")).to_series().to_list()
print(f"Removed {len(insufficient_regions)} regions")

# group the cases to the newly defined regions (# no filtering after 2020!)
cases_wo_filtering = (
    cases
    # aggregate to regions
    .with_columns(pl.col(f'{region}').replace_strict(region_cluster_map).alias('largest_sampling_effort_id'))
    .group_by(["date", 'largest_sampling_effort_id'])
    .agg(agg_exprs)
    .with_columns([
        pl.when(pl.col(f"{c}_count") == 0)
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in agg_cols
    ])
    # count serotyping effort
    .with_columns(
        DENV_serotyped_count=(
            pl.when(pl.sum_horizontal("^DENV_[1-4]$") == 0)
            .then(None)
            .otherwise(pl.sum_horizontal("^DENV_[1-4]$"))
        )
    )
    .drop([f"{c}_count" for c in agg_cols])
    .sort(["date", 'largest_sampling_effort_id'])
).to_pandas()

# group the cases to the newly defined regions
cases_w_filtering = (
    cases
    # Filter out any regions present in the insufficient_regions list
    .filter(~pl.col("CD_RGINT").is_in(insufficient_regions))
    # aggregate to regions
    .with_columns(pl.col(f'{region}').replace_strict(region_cluster_map).alias('largest_sampling_effort_id'))
    .group_by(["date", 'largest_sampling_effort_id'])
    .agg(agg_exprs)
    .with_columns([
        pl.when(pl.col(f"{c}_count") == 0)
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in agg_cols
    ])
    # count serotyping effort
    .with_columns(
        DENV_serotyped_count=(
            pl.when(pl.sum_horizontal("^DENV_[1-4]$") == 0)
            .then(None)
            .otherwise(pl.sum_horizontal("^DENV_[1-4]$"))
        )
    )
    .drop([f"{c}_count" for c in agg_cols])
    .sort(["date", 'largest_sampling_effort_id'])
).to_pandas()

# visualise them per cluster
dates = cases['date'].unique()
for cluster_id in output['largest_sampling_effort_id'].unique():

    fig,ax=plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(8.3, 11.7))

    fig.suptitle(f"cluster: {cluster_id}")

    # wo filtering
    ax[0,0].plot(dates, cases_wo_filtering.loc[cases_wo_filtering['largest_sampling_effort_id'] == cluster_id, 'DENV_serotyped_count'], color='black')
    ax[0,0].set_title("Without filtering")
    ax[0,0].set_ylabel("Total serotyped (-)")
    # w filtering
    ax[0,1].plot(dates, cases_w_filtering.loc[cases_w_filtering['largest_sampling_effort_id'] == cluster_id, 'DENV_serotyped_count'], color='black')
    ax[0,1].set_title(f"With filtering (threshold: {threshold})")

    for i in range(1,5):
        # wo filtering
        ax[i,0].plot(dates, cases_wo_filtering.loc[cases_wo_filtering['largest_sampling_effort_id'] == cluster_id, f'DENV_{i}'] / cases_wo_filtering.loc[cases_wo_filtering['largest_sampling_effort_id'] == cluster_id, 'DENV_serotyped_count'] * 100, marker='o', markersize=2, linewidth=1, color='black')
        ax[i,0].set_ylabel(f"DENV_{i} (%)")
        # w filtering
        ax[i,1].plot(dates, cases_w_filtering.loc[cases_w_filtering['largest_sampling_effort_id'] == cluster_id, f'DENV_{i}'] / cases_w_filtering.loc[cases_w_filtering['largest_sampling_effort_id'] == cluster_id, 'DENV_serotyped_count'] * 100, marker='o', markersize=2, linewidth=1, color='black')
        
    plt.tight_layout()
    os.makedirs(f'../../data/interim/nearest-largest-sampling-effort/cases/', exist_ok=True)
    plt.savefig(f'../../data/interim/nearest-largest-sampling-effort/cases/cluster_{cluster_id}.pdf')
