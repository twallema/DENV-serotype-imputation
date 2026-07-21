
import os
import polars as pl
import pandas as pd
import numpy as np
import pymc as pm
import arviz
from patsy import dmatrix
import geopandas as gpd
import pytensor.tensor as pt
import matplotlib.pyplot as plt

# spatial aggregation: 'rgint' (130 intermediate regions) ONLY
region_filename = 'rgint'   # NOTE: script intended to work with 'rgint'
region = 'CD_RGINT'         # NOTE: script intended to work with 'CD_RGINT'

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
    # aggregate to clusters
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
).to_pandas()

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

# build spline basis
t = np.arange(n_months)
X = np.asarray(
    dmatrix(
        f"bs(t, df={int(np.round(n_months/24))}, degree=3, include_intercept=True)",
        {"t": t},
    )
)
n_basis = X.shape[1]
X_pt = pt.constant(X)

# construct coordinates
coords = {
    "date": cases['date'].unique(),
    f"{region}": cases[f"{region}"].unique(),
    "serotype": [1, 2, 3, 4],
    "spline_basis": np.arange(n_basis),
}

# build pymc model
with pm.Model(coords=coords) as model:

    # splined latent state
    beta = pm.Normal("beta", 0, 1, shape=(n_regions, n_serotypes-1, n_basis))
    theta_partial = pm.Deterministic("theta_log", pt.einsum("tb,rsb->trs", X, beta))
    theta_log = pt.concatenate([theta_partial, pt.zeros((n_months,n_regions,1))], axis=2)

    # softmax to obtain latent serotype distribution
    p = pm.Deterministic("p", pm.math.softmax(theta_log, axis=2), dims=("date", f"{region}", "serotype"))

    # Overdispersion models
    ## Time-independent hierarchical overdispersion (per cluster)
    d_region_hierarch = pm.HalfNormal("d_region_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
    d_region = pm.HalfNormal("d_region", sigma=d_region_hierarch, dims=f"{region}")
    phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_region, 1e-12))[None, :], n_months, axis=0), dims=("date", f"{region}"))
    alpha = phi[:, :, None] * p # Broadcast phi over serotypes

    # --- Observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial, dims=("date", f"{region}", "serotype"))

# NUTS
draws=100
with model:
    trace = pm.sample(draws, tune=250, target_accept=0.8, chains=4, cores=4, init='adapt_diag', progressbar=True, idata_kwargs={'log_likelihood':True})

# Traceplot
variables2plot = ['d_region_hierarch', 'd_region']

# Save traces
os.makedirs(f'fig/trace', exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'fig/trace/trace-{var}_typing-effort-model.pdf')
    plt.close()
