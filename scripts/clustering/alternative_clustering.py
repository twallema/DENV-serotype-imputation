"""
In this script, I impute the serotype data for areas with a large sampling effort (LSE area) only. I then compute the log-likelihood of all non-LSE areas using the imputed serotype distribution of the LSE areas. This gives a measure of serotype proximity.
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
from patsy import dmatrix
import arviz
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.stats import dirichlet_multinomial
from glasbey import create_palette
from matplotlib.colors import ListedColormap


abs_dir = os.path.dirname(__file__)


#####################
## Script settings ##
#####################

region = 'CD_RGINT'
spatial_aggregation = 'rgint'

n_draw = 25
n_tune = 25
n_chains = 4

ID = 'test'

start_month_season = 9 
median_serotyped_case_threshold = 0

# Make output directory
output_folder = os.path.join(abs_dir, f'../../data/interim/alternative_clustering/{ID}/clusters/')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


########################
## Data preprocessing ##
########################

# Load case data
cases = pl.scan_parquet("../../data/interim/datasus_DENV-linelist/DENV-1999_2026-month-mun-no_diagnostics.parquet").collect()

# Load geodata
geography = gpd.read_parquet(os.path.join(abs_dir, "../../data/interim/geographic-dataset.parquet"))
geography = geography.to_crs('EPSG:5880')
gdf_states = geography.dissolve(by='CD_UF')
gdf_regions = geography.dissolve(by=f'{region}').reset_index()
regions = gdf_regions[f"{region}"].unique().astype(int)

# Make a mapping from municipalities to regions
muncipality_region_map = geography[['CD_MUN', f'{region}']].set_index('CD_MUN').to_dict()[f'{region}']

# Convert case data to regions
cases = (
    cases
    .with_columns(
        pl.col("CD_MUN")
        .replace(muncipality_region_map)
        .alias(f"{region}")
    )
    .group_by(["date", f"{region}"])
    .agg(
        pl.col([f"DENV_{i}" for i in range(1, 5)] + ['DENV_total']
    ).sum(),
    )
    .with_columns(
        pl.sum_horizontal([f"DENV_{i}" for i in range(1, 5)]
    ).alias("N_typed")
    )
    .sort(["date", f"{region}"])
)
cases_regions = cases

# Load nearest largest sampling effort
good_regions_quality = pd.read_csv(f"../../data/interim/DTW-MDS-embeddings/serotypes/{region}/serotype_trajectory_quality_{spatial_aggregation}.csv")
good_regions_quality = good_regions_quality.loc[good_regions_quality["has_quality"] == 1, f"{region}"].tolist()

# Filter the case data for the high quality sampling areas
cases_filtered = cases.filter(pl.col(f"{region}").is_in(good_regions_quality))

# If high quality sampling areas touch they must be functionally joined

## Select the high-quality regions
gdf_good = gdf_regions[gdf_regions[f"{region}"].isin(good_regions_quality)].copy()
gdf_good = gdf_good.reset_index(drop=True)
## Find which high-quality regions touch
touches = gpd.sjoin(gdf_good[["geometry"]], gdf_good[["geometry"]], predicate="touches", how="inner")
## Remove self-pairs
touches = touches[touches.index != touches["index_right"]]
## Find connected components using union-find
n = len(gdf_good)
parent = np.arange(n)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    x_root = find(x)
    y_root = find(y)

    if x_root != y_root:
        parent[y_root] = x_root

for i, j in zip(touches.index, touches["index_right"]):
    union(i, j)

## Compress component labels
roots = np.array([find(i) for i in range(n)])
unique_roots, component_id = np.unique(roots, return_inverse=True)
gdf_good["LSE_region"] = component_id

## Create the merged geometry
gdf_high_quality = (
    gdf_good
    .dissolve(
        by="LSE_region",
        as_index=False,
        aggfunc={f"{region}": list},
    )
    .sort_values("LSE_region")
    .reset_index(drop=True)
)
gdf_high_quality["n_regions"] = gdf_high_quality[f"{region}"].str.len()

## Aggregate the cases
region_to_high_quality = dict(zip(gdf_good[f"{region}"],gdf_good["LSE_region"]))

cases_high_quality = (
    cases_filtered
    .with_columns(
        pl.col(f"{region}")
        .replace(region_to_high_quality)
        .alias("LSE_region")
    )
    .group_by(["date", "LSE_region"])
    .agg(
        pl.col([f"DENV_{i}" for i in range(1, 5)]).sum(),
        pl.col("DENV_total").sum(),
        pl.col("N_typed").sum(),
    )
    .sort(["date", "LSE_region"])
)


#################################
## Imputation of the LSE areas ##
#################################

# indices and lengths
dates = cases_high_quality["date"].unique().sort().to_numpy()
LSE_regions = cases_high_quality["LSE_region"].unique().sort().to_numpy()
n_months = len(dates)
n_LSE_regions = len(LSE_regions)
n_serotypes = 4

# Build the observed matrices needed for pyMC
N = cases_high_quality.pivot(index="date", on="LSE_region", values="N_typed").to_pandas().set_index('date').to_numpy().astype(int)                # (n_months, n_LSE_regions)
Y = cases_high_quality.sort(["date", "LSE_region"]).select([f"DENV_{i}" for i in range(1, 5)]).to_numpy().reshape(len(dates), len(LSE_regions), n_serotypes)    # (n_months, n_LSE_regions, n_serotypes)

# build a spline basis
t = np.arange(n_months)
X = np.asarray(
    dmatrix(
        f"bs(t, df={int(np.round(n_months/9))}, degree=3, include_intercept=True)",
        {"t": t},
    )
)
n_basis = X.shape[1]
X_pt = pt.constant(X)

# construct model coordinates
coords = {
    "date": dates,
    "LSE_region": LSE_regions,
    "serotype": np.array([1, 2, 3, 4]),
    "spline_basis": np.arange(n_basis),
}

# build pymc imputation model
with pm.Model(coords=coords) as model:

    # spline coefficients
    sigma_beta = pm.HalfNormal("sigma_beta", 1)
    beta = pm.Normal("beta", 0, sigma_beta, shape=(n_LSE_regions, n_serotypes-1, n_basis))

    # build splined latent state 
    theta_log = pm.Deterministic("theta_log", pt.concatenate([pt.einsum("tb,rsb->trs", X, beta), pt.zeros((n_months,n_LSE_regions,1))], axis=2), dims=("date", "LSE_region", "serotype"))

    # softmax splined latent state to obtain latent serotype distribution
    p = pm.Deterministic("p", pm.math.softmax(theta_log, axis=2), dims=("date", "LSE_region", "serotype"))

    # overdispersion model
    ## time-independent hierarchical overdispersion (per region)
    d_region_hierarch = pm.HalfNormal("d_region_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
    d_region = pm.HalfNormal("d_region", sigma=d_region_hierarch, dims="LSE_region")
    phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_region, 1e-12))[None, :], n_months, axis=0), dims=("date", "LSE_region"))
    alpha = phi[:, :, None] * p # Broadcast phi over serotypes

    # observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N, observed=Y, dims=("date", "LSE_region", "serotype"))

# NUTS
with model:
    trace = pm.sample(n_draw, tune=n_tune, target_accept=0.8, chains=n_chains, cores=n_chains, init='adapt_diag', progressbar=True)

# save traces
variables2plot = ['sigma_beta', 'd_region_hierarch', 'd_region']
os.makedirs(os.path.join(output_folder, f'imputation_model/trace'), exist_ok=True)
for var in variables2plot:
    arviz.plot_trace_dist(trace, var_names=[var], compact=True, combined=True, kind='kde') 
    plt.savefig(os.path.join(output_folder, f'imputation_model/trace/trace-{var}_typing-effort-model.pdf'))
    plt.close()

# make a posterior predictive
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)


####################################
## Visualisation of the LSE areas ##
####################################

cases = cases.to_pandas()
cases_high_quality = cases_high_quality.to_pandas()

# # loop over clusters
# for region_id in LSE_regions:

#     fig = plt.figure(figsize=(8.3, 11.7/6*8))
#     fig.suptitle(f"Cluster: {region_id}")
#     gs = fig.add_gridspec(8, 2)

#     # map highlighting the region
#     ax = fig.add_subplot(gs[0, 1])
#     gdf_states.boundary.plot(ax=ax, linewidth=0.5, color="black")
#     gdf_regions.boundary.plot(ax=ax, linewidth=0.1, color="black", alpha=0.2)
#     gdf = gdf_high_quality.loc[gdf_high_quality["LSE_region"] == region_id]
#     gdf.plot(ax=ax, color="#d35052", edgecolor="none")
#     ax.set_axis_off()

#     # set up rows below to span columns
#     ax = []
#     ax.append(fig.add_subplot(gs[1, :]))

#     for r in range(2, 8):
#         ax.append(fig.add_subplot(gs[r, :], sharex=ax[0]))

#     for a in ax[:-1]:
#         plt.setp(a.get_xticklabels(), visible=False)


#     ax[0].plot(dates, cases_high_quality.loc[cases_high_quality["LSE_region"] == region_id, 'DENV_total'], color='black')
#     ax[0].set_ylabel("DENV cases (-)")

#     ax[1].plot(dates, cases_high_quality.loc[cases_high_quality["LSE_region"] == region_id, 'N_typed'], color='black')
#     ax[1].set_ylabel("Serotyped cases (-)")

#     for s in range(1,5):
#         ax[s+1].set_ylabel(f"DENV {s} (%)")
#         # data
#         ax[s+1].plot(dates, cases_high_quality.loc[cases_high_quality["LSE_region"] == region_id, f'DENV_{s}'].values / cases_high_quality.loc[cases_high_quality["LSE_region"] == region_id, 'N_typed'].values * 100, marker='o', markersize=2, linewidth=1, color='black')
#         # model
#         ax[s+1].plot(dates, trace.posterior['p'].median(dim=['chain', 'draw']).sel({'serotype': s, "LSE_region": region_id}).values * 100, color='red')
#         ax[s+1].fill_between(dates,
#                                 trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.025).sel({'serotype': s, "LSE_region": region_id}).values * 100,
#                                 trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.975).sel({'serotype': s, "LSE_region": region_id}).values * 100,
#                                 color='red', alpha=0.1
#                             )
#         ax[s+1].fill_between(dates,
#                                 trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.25).sel({'serotype': s, "LSE_region": region_id}).values * 100,
#                                 trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.75).sel({'serotype': s, "LSE_region": region_id}).values * 100,
#                                 color='red', alpha=0.2
#                             )
        
#     ax[-1].stackplot(dates, [trace.posterior['p'].mean(dim=['chain', 'draw']).sel({'serotype': serotype, "LSE_region": region_id}).values * 100 for serotype in range(1,5)], labels=['1', '2', '3', '4'], colors=['black', 'red', 'green', 'blue'], alpha=0.9)
#     ax[-1].set_ylabel(f"Serotype distribution (%)")

#     plt.tight_layout()
#     os.makedirs(os.path.join(output_folder, 'imputation_model/posterior_predictive'), exist_ok=True)
#     plt.savefig(os.path.join(output_folder, f'imputation_model/posterior_predictive/cluster_{region_id}.pdf'))
#     plt.close()


################################
## Log likelihood computation ##
################################

# Pre-allocate a matrix of size (n_regions x n_LSE_regions)
ll = np.zeros([len(gdf_regions), n_LSE_regions])

# Loop over all regions (including the high quality ones)
for i, region_id in enumerate(regions):

    # Select the total number of typed cases N_typed and observed cases per serotyped Y
    N = cases.loc[cases[f'{region}'] == region_id, 'N_typed'].values
    Y = cases.loc[cases[f'{region}'] == region_id, ('DENV_1', 'DENV_2', 'DENV_3', 'DENV_4')].values

    ## Loop over all high quality regions
    for j, LSE_region_id in enumerate(LSE_regions):

        ## Select the latent serotype distribution and overdispersion coefficient to construct a (n_months x n_serotypes)
        p = np.squeeze(trace.posterior['p'].median(dim=['chain', 'draw']).sel({"LSE_region": LSE_region_id}).values)
        phi = trace['posterior']['phi'].mean(dim=['chain','draw']).sel({"LSE_region": LSE_region_id}).values
        alpha = phi[:, None] * p

        ## Compute the likelihood of the datapoints
        logp = dirichlet_multinomial.logpmf(x=Y, alpha=alpha, n=N)

        ## Sum them and store them in the pre-allocated matrix
        ll[i,j] = sum(logp)

# Store the closest LSE for every region
gdf_regions['closest_LSE_id'] = np.argmax(ll, axis=1)


#########################################################
## Filter out regions with extremely sparse serotyping ##
#########################################################

# Load nearest largest sampling effort
nearest_largest_sampling_effort = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/nearest-largest-sampling-effort/nearest-largest-sampling-effort_{spatial_aggregation}.csv'))

# load the overland travel time
travel_time_matrix = pd.read_csv(f'../../data/interim/travel_time_matrices/travel-time_car_mean_{spatial_aggregation}.csv', index_col=0)
travel_time_matrix.columns = travel_time_matrix.columns.astype(int)


# determine what areas serotyping is too sparse

## append a "season" label (September of year X -> September of year X+1)
cases_season = (
    cases_regions 
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
    .sort(f"{region}")
)

# enforce threshold
sparse_regions = cases_season_stats.filter(pl.col("median") <= median_serotyped_case_threshold)[f'{region}'].to_list()

# update closest_LSE_id with lowest travel time for sparse areas
manual_assigment_dictionary = {
    'CD_RGINT': {1302: 1301, 1303: 1301, 1304: 1301, 1201: 1301, 1202: 1301},   
    'CD_RGI': {130002: 130001, 130003: 130001, 130005: 130001, 130006: 130001, 130007: 130001, 130010: 130001, 130011: 130001, 150020: 150001}
    }

for sparse_region in sparse_regions:
    try:
        closest_high_quality = travel_time_matrix.loc[sparse_region, good_regions_quality].idxmin()
    except:
        closest_high_quality = manual_assigment_dictionary[f'{region}'][sparse_region]

    gdf_regions.loc[gdf_regions[f'{region}'] == sparse_region, 'closest_LSE_id'] =  gdf_high_quality.loc[gdf_high_quality[f"{region}"].apply(lambda x: closest_high_quality in x), "LSE_region"].iloc[0]

print(f"Number of sparse regions: {len(sparse_regions)} / {len(regions)}")

#####################################
## Visualise lowest log likelihood ##
#####################################

# Visualise on a map 
fig, ax = plt.subplots()

glasbey_cmap = ListedColormap(create_palette(palette_size=len(gdf_high_quality)))

gdf_states.boundary.plot(ax=ax, linewidth=0.5, color="black")   # state boundaries

gdf_regions.plot(
    column=f"closest_LSE_id",          # color regions by cluster label
    categorical=True,
    cmap = glasbey_cmap,
    linewidth=0.2,
    edgecolor="grey",
    legend=False,
    ax=ax,
)

gdf_high_quality.plot(
    ax=ax,
    color="grey",
    edgecolor="black",
    hatch="/////",
    linewidth=0.5,
    alpha=0.6  # Slight transparency softens the grey
)
ax.axis("off")
plt.savefig(os.path.join(output_folder, f'nearest_LSE_{spatial_aggregation}.png'))
plt.close()

