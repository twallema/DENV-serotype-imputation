import os
import polars as pl
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

ranking = (
    cases_season_stats
    .with_columns([
        # Percentile ranks (0.0 to 1.0)
        (pl.col("median").rank("dense") / pl.count()).alias("rank_median"),
        (pl.col("qcd").rank("dense", descending=True) / pl.count()).alias("rank_qcd"), # lower QCD is better
        (pl.col("skew").fill_null(99).rank("dense", descending=True) / pl.count()).alias("rank_skew"), # lower skew is better
    ])
    .with_columns(
        # Weighted Composite Score (adjust weights if median matters more than skew)
        serotyping_index = (
            1.0 * pl.col("rank_median") + 
            0.0 * pl.col("rank_qcd") + 
            0.0 * pl.col("rank_skew")
        )
    )
    .sort("serotyping_index", descending=True)
)

good_regions_filtered = ranking[:len(good_regions_quality[good_regions_quality['has_quality'] == 1])]


####################
## Visualise them ##
####################

geography_states = geography.dissolve(by='CD_UF')

geography_regions = geography.dissolve(by=f'{region}').reset_index()

geography_regions.loc[geography_regions[f"{region}"].isin(good_regions_filtered.select(pl.col(f"{region}")).to_series().to_list()), 'cluster'] = 'quantitative'
geography_regions.loc[geography_regions[f"{region}"].isin(good_regions_quality.loc[good_regions_quality['has_quality']==1, f"{region}"]), 'cluster'] = 'qualitative'
geography_regions.loc[((geography_regions[f"{region}"].isin(good_regions_filtered.select(pl.col(f"{region}")).to_series().to_list())) & (geography_regions[f"{region}"].isin(good_regions_quality.loc[good_regions_quality['has_quality']==1, f"{region}"]))), 'cluster'] =  'overlap'
geography_regions['cluster'] = geography_regions['cluster'].fillna('other')

fig, ax = plt.subplots()
geography_states.boundary.plot(ax=ax, linewidth=0.5, color="black")   # state boundaries
geography_regions.plot(
    column="cluster",          # color regions by cluster label
    categorical=True,
    linewidth=0.2,
    edgecolor="grey",
    legend=True,
    ax=ax,
    legend_kwds={'fontsize': 4, 'ncol': 2, 'loc': 'lower right', 'markerscale': 0.4}
)
ax.set_title(f"Median number of serotyped cases per season", fontsize=12)
ax.axis("off")
os.makedirs(f'../../data/interim/nearest-largest-sampling-effort/', exist_ok=True)
plt.savefig(f'../../data/interim/nearest-largest-sampling-effort/qualitative-quantitative-comparison_{region_filename}.png', dpi=600)
plt.close()


#####################
## Turn into a map ##
#####################

geography_regions = geography.dissolve(by=f'{region}').reset_index()

# merge the good quality regions
gdf = geography_regions.merge(good_regions_quality, on="CD_RGINT")

# STEP 1: Merge contiguous quality regions into clusters
quality_gdf = gdf[gdf["has_quality"] == 1].copy()
## Dissolve touching quality regions into unified geometries to assign Cluster IDs
clusters = (
    quality_gdf.dissolve()
    .explode(index_parts=False)
    .reset_index(drop=True)
)
clusters["cluster_id"] = clusters.index + 1
## Map cluster_id back to every individual quality region
quality_regions = gpd.sjoin(
    quality_gdf,
    clusters[["cluster_id", "geometry"]],
    how="inner",
    predicate="intersects"
)[["CD_RGINT", "cluster_id", "geometry"]].drop_duplicates(subset=["CD_RGINT"])

# STEP 2 & 3: Calculate centroids & find nearest quality centroid
## Calculate centroids for individual quality regions
quality_centroids = quality_regions.copy()
quality_centroids["geometry"] = quality_centroids.geometry.centroid
## Calculate centroids for non-quality regions
non_quality_gdf = gdf[gdf["has_quality"] == 0].copy()
non_quality_centroids = non_quality_gdf[["CD_RGINT", "geometry"]].copy()
non_quality_centroids["geometry"] = non_quality_centroids.geometry.centroid
## Compute nearest centroid distance between non-quality regions and quality regions
nearest_matches = gpd.sjoin_nearest(
    non_quality_centroids,
    quality_centroids[["CD_RGINT", "cluster_id", "geometry"]],
    how="left"
)

# STEP 4: Combine quality & non-quality regions into final table
quality_final = quality_regions[["CD_RGINT", "cluster_id"]]
non_quality_final = (
    nearest_matches[["CD_RGINT_left", "cluster_id"]]
    .rename(columns={"CD_RGINT_left": "CD_RGINT"})
)
result_df = (
    pd.concat([quality_final, non_quality_final], ignore_index=True)
    .drop_duplicates(subset=["CD_RGINT"])
    .sort_values(by="CD_RGINT")
    .reset_index(drop=True)
)
result_df["cluster_id"] = pd.factorize(result_df["cluster_id"])[0]

geography_regions = geography_regions.merge(result_df, on="CD_RGINT")
geography_regions = geography_regions.merge(good_regions_quality, on="CD_RGINT")

# STEP 5: Visualise on a map 
glasbey_cmap = ListedColormap(create_palette(palette_size=len(good_regions_quality[good_regions_quality['has_quality'] == 1])))

geography_states = geography.dissolve(by='CD_UF')

fig, ax = plt.subplots()
geography_states.boundary.plot(ax=ax, linewidth=0.5, color="black")   # state boundaries

geography_regions.plot(
    column="cluster_id",          # color regions by cluster label
    categorical=True,
    cmap = glasbey_cmap,
    linewidth=0.2,
    edgecolor="grey",
    legend=True,
    ax=ax,
    legend_kwds={'fontsize': 4, 'ncol': 2, 'loc': 'lower right', 'markerscale': 0.4}
)

quality_regions = geography_regions[geography_regions['has_quality'] == 1]
quality_regions.plot(
    ax=ax,
    color="grey",
    edgecolor="black",
    hatch="////",
    linewidth=0.5,
    alpha=0.4  # Slight transparency softens the grey
)
os.makedirs(f'../../data/interim/nearest-largest-sampling-effort/', exist_ok=True)
plt.savefig(f'../../data/interim/nearest-largest-sampling-effort/nearest-largest-sampling-effort_{region_filename}.png', dpi=600)
plt.close()

#################
## Save output ##
#################

output = geography_regions[[f'{region}', 'cluster_id']]
output = output.rename(columns={'cluster_id': 'largest_sampling_effort'})
output.to_csv(f'../../data/interim/nearest-largest-sampling-effort/nearest-largest-sampling-effort_{region_filename}.csv', index=False)


#######################################################################################
## Visualise data in these clusters with and without removal of low-data states data ##
#######################################################################################

# make a mapping from f'{region}' to the newly defined closest-largest-sampling-effort regions
region_cluster_map = output.set_index(f'{region}').to_dict()['largest_sampling_effort']

# get regions with insufficient data
threshold = 2
insufficient_regions = cases_season_stats.filter(pl.col("median") <= threshold).select(pl.col(f"{region}")).to_series().to_list()
print(f"Removed {len(insufficient_regions)} regions")

# group the cases to the newly defined regions (# no filtering after 2020!)
cases_wo_filtering = (
    cases
    # aggregate to regions
    .with_columns(pl.col(f'{region}').replace_strict(region_cluster_map).alias('largest_sampling_effort'))
    .group_by(["date", 'largest_sampling_effort'])
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
    .sort(["date", 'largest_sampling_effort'])
).to_pandas()

# group the cases to the newly defined regions
cases_w_filtering = (
    cases
    # Filter out any regions present in the insufficient_regions list
    .filter(~pl.col("CD_RGINT").is_in(insufficient_regions))
    # aggregate to regions
    .with_columns(pl.col(f'{region}').replace_strict(region_cluster_map).alias('largest_sampling_effort'))
    .group_by(["date", 'largest_sampling_effort'])
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
    .sort(["date", 'largest_sampling_effort'])
).to_pandas()

# visualise them per cluster
dates = cases['date'].unique()
for cluster_id in output['largest_sampling_effort'].unique():

    fig,ax=plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(8.3, 11.7))

    fig.suptitle(f"cluster: {cluster_id}")

    # wo filtering
    ax[0,0].plot(dates, cases_wo_filtering.loc[cases_wo_filtering['largest_sampling_effort'] == cluster_id, 'DENV_serotyped_count'], color='black')
    ax[0,0].set_title("Without filtering")
    ax[0,0].set_ylabel("Total serotyped (-)")
    # w filtering
    ax[0,1].plot(dates, cases_w_filtering.loc[cases_w_filtering['largest_sampling_effort'] == cluster_id, 'DENV_serotyped_count'], color='black')
    ax[0,1].set_title(f"With filtering (threshold: {threshold})")

    for i in range(1,5):
        # wo filtering
        ax[i,0].plot(dates, cases_wo_filtering.loc[cases_wo_filtering['largest_sampling_effort'] == cluster_id, f'DENV_{i}'] / cases_wo_filtering.loc[cases_wo_filtering['largest_sampling_effort'] == cluster_id, 'DENV_serotyped_count'] * 100, marker='o', markersize=2, linewidth=1, color='black')
        ax[i,0].set_ylabel(f"DENV_{i} (%)")
        # w filtering
        ax[i,1].plot(dates, cases_w_filtering.loc[cases_w_filtering['largest_sampling_effort'] == cluster_id, f'DENV_{i}'] / cases_w_filtering.loc[cases_w_filtering['largest_sampling_effort'] == cluster_id, 'DENV_serotyped_count'] * 100, marker='o', markersize=2, linewidth=1, color='black')
        
    plt.tight_layout()
    os.makedirs(f'../../data/interim/nearest-largest-sampling-effort/cases/', exist_ok=True)
    plt.savefig(f'../../data/interim/nearest-largest-sampling-effort/cases/cluster_{cluster_id}.pdf')
