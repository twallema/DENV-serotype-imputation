
import os
import polars as pl
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import arviz
import pymc as pm
import pytensor.tensor as pt
from patsy import dmatrix

import seaborn as sns
from tslearn.metrics import cdist_dtw
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from glasbey import create_palette
from matplotlib.colors import ListedColormap

##############
## Settings ##
##############

# spatial aggregation: 'rgint' (130 intermediate regions) ONLY
region_filename = 'rgint'   # NOTE: script intended to work with 'rgint'
region = 'CD_RGINT'         # NOTE: script intended to work with 'CD_RGINT'

# sampler
n_chains = 4
n_tune = 50
n_draw = 100

start_month_season = 9

# dtw
n_dtw_clusters = [2,3,4,5,6,7,8]

###############
## Data prep ##
###############

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
    #.with_columns(DENV_serotyped_count=pl.sum_horizontal("^DENV_[1-4]_count$"))
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

########################################
## Cluster based on serotyping effort ##
########################################

# compute (median) serotyped cases per season --> cluster and visualise on map
## append a "season" label (September of year X -> September of year X+1)
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

## compute median across seasons per cluster
cases_season = (
    cases_season
    .group_by(f"{region}")
    .agg(pl.col("total_serotyped").median().alias("median_serotyped"))
    .sort(f"{region}")
)

# cluster and plot on map
features = cases_season.select("median_serotyped").to_numpy()
Z = linkage(features, method="ward")

geography_states = geography.dissolve(by='CD_UF')

geography_regions = geography.dissolve(by=f'{region}').reset_index()

for n_clusters in n_dtw_clusters:

    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    cases_per_cluster = cases_season.with_columns(pl.Series(name="cluster_id", values=clusters)).group_by('cluster_id').agg(pl.col('median_serotyped').mean()).sort('cluster_id')

    label_map = {row["cluster_id"]: f"Cluster {row['cluster_id']} ({row['median_serotyped']:.1f})" for row in cases_per_cluster.iter_rows(named=True)}
    
    geography_regions["median_serotyped_cluster"] = [label_map[c] for c in clusters]

    glasbey_cmap = ListedColormap(create_palette(palette_size=n_clusters))

    fig, ax = plt.subplots()
    geography_states.boundary.plot(ax=ax, linewidth=0.5, color="black")   # state boundaries
    geography_regions.plot(
        column="median_serotyped_cluster",          # color regions by cluster label
        cmap = glasbey_cmap,
        categorical=True,
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        ax=ax,
        legend_kwds={'fontsize': 4, 'ncol': 2, 'loc': 'lower right', 'markerscale': 0.4}
    )
    ax.set_title(f"Median number of serotyped cases per season", fontsize=12)
    ax.axis("off")
    os.makedirs(f'fig/serotyping_effort', exist_ok=True)
    plt.savefig(f'fig/serotyping_effort/median_clusters_{n_clusters}.pdf')
    plt.close()


######################
## Imputation model ##
######################

cases = cases.to_pandas()

# Total number of serotyped cases
N_typed = cases.pivot(index="date", columns=f"{region}", values="DENV_serotyped_count").fillna(0).to_numpy().astype(int) # (n_months, n_regions)

# Number of cases per DENV serotype
Y_list = []
for col in ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']:
    Y_mat = cases.pivot(index="date", columns=f"{region}", values=col).to_numpy()
    Y_list.append(Y_mat)
Y_multinomial = np.stack(Y_list, axis=2).astype(int)    # (n_months, n_regions, n_serotypes)

# Lengths
n_months = Y_multinomial.shape[0]
n_regions = Y_multinomial.shape[1]
n_serotypes = Y_multinomial.shape[2]

# build an adjacency matrix
from libpysal.weights import Queen
regions = (
    geography[["CD_RGINT", "geometry"]]
    .dissolve(by="CD_RGINT", as_index=False)
    .sort_values("CD_RGINT")
    .reset_index(drop=True)
)
queen = Queen.from_dataframe(regions)
adj = queen.full()[0].astype(int)
np.fill_diagonal(adj, 0)

I = pt.eye(len(regions))
W = pt.as_tensor_variable(adj)
D = pt.diag(pt.sum(W, axis=1))

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
    "date": cases['date'].unique(),
    f"{region}": cases[f"{region}"].unique(),
    "serotype": np.array([1, 2, 3, 4]),
    "spline_basis": np.arange(n_basis),
}

# build pymc imputation model
with pm.Model(coords=coords) as model:

    # spatially correlated spline coefficients
    psi = pm.Beta("psi", 3, 3)
    sigma_beta = pm.HalfNormal("sigma_beta", 1)
    Q = (1 - psi) * I + psi * (D - W)
    L_Q = pt.linalg.cholesky(Q)
    L_cov = pt.linalg.solve(L_Q, I)

    beta_raw = pm.Normal("beta_raw", 0, 1, shape=(n_basis, n_serotypes - 1, n_regions))
    beta_corr = sigma_beta * pt.einsum("ij,bsj->bsi", L_cov, beta_raw)
    beta = pm.Deterministic("beta", beta_corr.dimshuffle(2, 1, 0))
    
    # build splined latent state 
    theta_log = pm.Deterministic("theta_log", pt.concatenate([pt.einsum("tb,rsb->trs", X, beta), pt.zeros((n_months,n_regions,1))], axis=2), dims=("date", f"{region}", "serotype"))

    # softmax splined latent state to obtain latent serotype distribution
    p = pm.Deterministic("p", pm.math.softmax(theta_log, axis=2), dims=("date", f"{region}", "serotype"))

    # overdispersion model
    ## time-independent hierarchical overdispersion (per region)
    d_region_hierarch = pm.HalfNormal("d_region_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
    d_region = pm.HalfNormal("d_region", sigma=d_region_hierarch, dims=f"{region}")
    phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_region, 1e-12))[None, :], n_months, axis=0), dims=("date", f"{region}"))
    alpha = phi[:, :, None] * p # Broadcast phi over serotypes

    # observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial, dims=("date", f"{region}", "serotype"))

# NUTS
with model:
    trace = pm.sample(n_draw, tune=n_tune, target_accept=0.8, chains=n_chains, cores=n_chains, init='adapt_diag', progressbar=True)

# save traces
variables2plot = ['sigma_beta', 'psi', 'd_region_hierarch', 'd_region']
os.makedirs(f'fig/trace', exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'fig/trace/trace-{var}_typing-effort-model.pdf')
    plt.close()

# make a posterior predictive
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)

#################################################
## DTW imputed serotype trajectories + cluster ##
#################################################

# DTW the latent serotype trajectory
theta_log = trace.posterior['theta_log'].mean(dim=['chain', 'draw'])
data_3d = theta_log.transpose("CD_RGINT", "date", "serotype").values
distance = cdist_dtw(data_3d, global_constraint="sakoe_chiba", sakoe_chiba_radius=12)

# cluster it
Z = linkage(squareform(distance, checks=False), method='average')
geography_regions = geography.dissolve(by=f'{region}').reset_index()

glasbey_cmap = ListedColormap(create_palette(palette_size=max(n_dtw_clusters)))

for n_clusters in n_dtw_clusters:

    clusters = fcluster(Z, n_clusters, criterion='maxclust')

    cases_per_cluster = cases_season.with_columns(pl.Series(name="cluster_id", values=clusters)).group_by('cluster_id').agg(pl.col('median_serotyped').mean()).sort('cluster_id')

    label_map = {row["cluster_id"]: f"Cluster {row['cluster_id']} ({row['median_serotyped']:.1f})" for row in cases_per_cluster.iter_rows(named=True)}
    
    geography_regions['dtw_cluster'] = [label_map[c] for c in clusters]

    sns.clustermap(distance, cmap='viridis', row_linkage=Z, col_linkage=Z, method='precomputed')
    os.makedirs(f'fig/dtw', exist_ok=True)
    plt.savefig(f'fig/dtw/dtw_clustermap.pdf')
    plt.close()

    # visualise clusters on a map
    geography_states = geography.dissolve(by='CD_UF')

    fig, ax = plt.subplots()
    geography_states.boundary.plot(ax=ax, linewidth=0.5, color="black")   # state boundaries
    geography_regions.plot(
        column="dtw_cluster",          # color regions by cluster label
        cmap = glasbey_cmap,
        categorical=True,
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        ax=ax,
        legend_kwds={'fontsize': 4, 'ncol': 2, 'loc': 'lower right', 'markerscale': 0.4}
    )
    ax.set_title(f"Serotype trajectory DTW distance", fontsize=12)
    ax.axis("off")
    plt.savefig(f'fig/dtw/dtw_clusters_{n_clusters}.pdf')
    plt.close()


##############################################
## Visualise imputed serotype trajectories  ##
##############################################

# visualise trajectories of serotype distribution
geography_states = geography.dissolve(by='CD_UF')
geography_regions = geography.dissolve(by=f'{region}').reset_index()

dates = posterior_predictive.posterior_predictive.coords['date'].values

for region_id in cases[f"{region}"].unique():

    N_typed = posterior_predictive.observed_data["Y_obs"].sum(dim="serotype").sel({f"{region}": region_id}).values
    
    fig = plt.figure(figsize=(8.3, 11.7))
    fig.suptitle(f"{region}: {region_id}")
    gs = fig.add_gridspec(7, 2)

    # map highlighting the region
    ax = fig.add_subplot(gs[0, 1])
    geography_states.boundary.plot(ax=ax, linewidth=0.5, color="black")
    geography_regions.boundary.plot(ax=ax, linewidth=0.1, color="black", alpha=0.2)
    gdf = geography_regions.loc[geography_regions[f'{region}'] == region_id]
    gdf.plot(ax=ax, color="#d35052", edgecolor="none")
    ax.set_axis_off()

    # set up rows below to span columns
    ax = []
    ax.append(fig.add_subplot(gs[1, :]))

    for r in range(2, 7):
        ax.append(fig.add_subplot(gs[r, :], sharex=ax[0]))

    for a in ax[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)

    # plot bottom timeseries
    ax[0].plot(dates, N_typed, color='black')
    ax[0].set_ylabel("No. serotyped (-)")

    for serotype in range(1,5):
        with np.errstate(invalid='ignore', divide='ignore'):
            ax[serotype].plot(dates, posterior_predictive.observed_data['Y_obs'].sel({'serotype': serotype, f"{region}": region_id}).values / N_typed * 100, marker='o', markersize=2, linewidth=1, color='black')

        ax[serotype].plot(dates, trace.posterior['p'].median(dim=['chain', 'draw']).sel({'serotype': serotype, f"{region}": region_id}).values * 100, color='red')
        ax[serotype].fill_between(dates,
                                  trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.025).sel({'serotype': serotype, f"{region}": region_id}).values * 100,
                                  trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.975).sel({'serotype': serotype, f"{region}": region_id}).values * 100,
                                  color='red', alpha=0.1
                                  )
        ax[serotype].fill_between(dates,
                                  trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.25).sel({'serotype': serotype, f"{region}": region_id}).values * 100,
                                  trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.75).sel({'serotype': serotype, f"{region}": region_id}).values * 100,
                                  color='red', alpha=0.2
                                  )
        ax[serotype].set_ylabel(f"DENV {serotype} (%)")
    
    ax[-1].stackplot(dates, [trace.posterior['p'].mean(dim=['chain', 'draw']).sel({'serotype': serotype, f"{region}": region_id}).values * 100 for serotype in range(1,5)], labels=['1', '2', '3', '4'], colors=['black', 'red', 'green', 'blue'], alpha=0.9)
    ax[-1].set_ylabel(f"Serotype distribution (%)")

    plt.tight_layout()
    os.makedirs(f'fig/posterior_predictive', exist_ok=True)
    plt.savefig(f'fig/posterior_predictive/{region}_{region_id}.pdf')
    plt.close()