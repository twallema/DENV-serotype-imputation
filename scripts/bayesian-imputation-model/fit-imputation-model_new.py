import os
import arviz
import argparse
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import redirect_stdout

import pytensor
import pytensor.tensor as pt
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.on_opt_error = "ignore"

# analysis startdate
start_year = 2001
end_year = 2024
assert start_year >= 2001, "demography data before 2001 is missing."

# helper function for argument parsing
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# arguments determine the model + data combo used to forecast
# How to run: python fit-model.py -ID test -p 2 -distance_matrix False
parser = argparse.ArgumentParser()
parser.add_argument("-ID", type=str, help="Identifier of the pipeline run.")
parser.add_argument("-spatial_aggregation", type=str, help="Spatial aggregation clustering was performed on.")
parser.add_argument("-chains", type=int, help="Number of parallel chains.", default=3)
args = parser.parse_args()

# assign to desired variables
spatial_aggregation = args.spatial_aggregation
chains = args.chains
ID = args.ID

# pipeline output folder
abs_dir = os.path.dirname(__file__) # make sure all referenced paths are relative to the lcoation of this file and not the terminal's pwd
output_folder = os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/bayesian-imputation-model_output/new_model/')
# check if output dir exists, if not, make it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


########################
## Preparing the data ##
########################

# Load left out spatial units
# >>>>>>>>>>>>>>>>>>>>>>>>>>>

validation_labels = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/clusters/validation_labels.csv')).squeeze()

# Load clusters
# >>>>>>>>>>>>>

clusters = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/clusters/clusters_{spatial_aggregation}.csv'))
region = clusters.columns.to_list()[0]

# Load mapping
# >>>>>>>>>>>>

mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))
mapping = mapping.merge(clusters[[region, 'cluster']], on=region, how='left')

# Compute demography in start_year per cluster
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

demo = pd.read_csv(os.path.join(abs_dir, f'../../data/raw/sprint_2025/datasus_population_2001_2024.csv'))
demo = demo.rename(columns={'geocode': 'CD_MUN'})
demo = demo.merge(mapping[['CD_MUN', 'cluster']], on='CD_MUN', how='left')
demo = demo.groupby(['cluster', 'year'], as_index=False)['population'].sum()

# Compute births and death rates per cluster
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

bd = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/IBGE_population-projections/IBGE_births-deaths_mun-estimated.csv'))
bd = bd.merge(mapping[['CD_MUN', 'cluster']], on='CD_MUN', how='left')
bd = bd.groupby(['year', 'cluster'], as_index=False).agg(estimated_births=('estimated_births', 'sum'),estimated_deaths=('estimated_deaths', 'sum'),population=('population', 'sum'))

# # plot actual versus estimated population
# bd['natural_increase'] = bd['estimated_births'] - bd['estimated_deaths']
# df_pop_2001 = (
#     demo
#     .query('year == 2001')
#     .loc[:, ['cluster', 'population']]
#     .rename(columns={'population': 'population_2001'})
# )
# bd = bd.merge(df_pop_2001, on='cluster', how='left')
# bd['estimated_population'] = (
#     bd['population_2001']
#     + bd.groupby('cluster')['natural_increase']
#          .transform(lambda x: x.cumsum().shift(fill_value=0))
# )
# for cluster in range(1,29):
#     fig,ax=plt.subplots()
#     ax.plot(demo[demo['cluster'] == cluster]['year'], demo[demo['cluster'] == cluster]['population'], color='black', label='actual')
#     ax.plot(bd[bd['cluster'] == cluster]['year'], bd[bd['cluster'] == cluster]['estimated_population'], color='red', label='estimated')
#     ax.legend()
#     plt.show()
#     plt.close()

# Adjacency matrix
# ~~~~~~~~~~~~~~~~

W = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/clusters/adjacency_matrix_{spatial_aggregation}.csv'), index_col=0).values

# Incidence data
# ~~~~~~~~~~~~~~

# Fetch incidence data
df = pd.read_csv(os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv'), parse_dates=['date'])

# 1. Check if all columns are present
sero_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4"]
required_cols = ["CD_MUN", "date", "DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]
assert all(col in df.columns for col in required_cols)

# 2. Sort for safety
df = df.sort_values(["CD_MUN", "date"]).reset_index(drop=True)

# 3. Take only from start_year to end_year
df = df[((df['date'] > datetime(start_year,1,1)) & (df['date'] <= datetime(end_year,12,31)))]

# 4. Remove within-sample validation municipalities
df.loc[df['CD_MUN'].isin(validation_labels.values), ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4', 'DENV_total'] ] = np.nan

# 5. Aggregate to the spatial clusters
# make right mapping
mapping = mapping[['CD_MUN', f'{region}']]
mapping = clusters.merge(mapping, on=f'{region}', how="left")
# do DENV_total first
df_with_mapping = df.merge(mapping[["CD_MUN", "cluster"]], on="CD_MUN", how="left")
## custom aggregation function
def nan_sum(series):
    if series.isna().all():
        return np.nan
    return series.sum(skipna=True)
## aggregate
agg_1 = (
    df_with_mapping
    .groupby(["date", "cluster"], as_index=False)
    .agg({"DENV_total": nan_sum})
)
# then aggregate DENV_1-->4
## follow same logic initially
agg_2 = (
    df_with_mapping
    .groupby(["date", "cluster"], as_index=False)
    .agg({
        "DENV_1": nan_sum,
        "DENV_2": nan_sum,
        "DENV_3": nan_sum,
        "DENV_4": nan_sum
    })
)
## but if you observe any serotype, others must be zero instead of Nan
mask = agg_2[["DENV_1", "DENV_2", "DENV_3", "DENV_4"]].notna().any(axis=1)
agg_2.loc[mask] = agg_2.loc[mask].fillna(0)
## merge both dataframes
df = agg_1.merge(agg_2)

# 6. Add number of serotyped cases
df["N_typed"] = df[sero_cols].sum(axis=1, skipna=False)           # if serotypes available --> sum them
df.loc[df[sero_cols].isna().all(axis=1), 'N_typed'] = np.nan      # if all serotypes are Nan --> N_typed = 0 --> Wait, I don't think this is appropriate.

# 7. Compute delta (typing fraction)
df["delta"] = df["N_typed"] / df["DENV_total"]
df['delta'] = df['delta'].where(df['N_typed'] > 0, np.nan) # When N_typed == 0, we don't know delta — mark as missing
df["delta"] = df["delta"].clip(lower=1e-12, upper=1 - 1e-12)

# 8. Compute year and month index
df["year"] = pd.to_datetime(df["date"]).dt.year
df["year_idx"] = df["year"] - df["year"].min()
df['month_idx'], _ = pd.factorize(df['date'])

# 9. Build PyMC arrays
# --- For DirichletMultinomial model ---
# Total number of typed cases
N_typed = df.pivot(index="date", columns="cluster", values="N_typed").to_numpy().astype(int)    # (n_months, n_clusters)
# Number of cases per DENV serotype
Y_list = []
for col in sero_cols:
    Y_mat = df.pivot(index="date", columns="cluster", values=col).to_numpy()
    Y_list.append(Y_mat)
Y_multinomial = np.stack(Y_list, axis=2).astype(int)    # (n_months, n_clusters, n_serotypes)

# --- For imunity model ---
# Total number of dengue cases
DENV_total = df.pivot(index="date", columns="cluster", values="DENV_total").to_numpy().astype(int)  # (n_months, n_clusters)
# Births (absolute) and death rate
df_expanded = df.merge(bd, on=["year", "cluster"], how="left")
df_expanded["estimated_births"] = df_expanded["estimated_births"] / 12
df_expanded["estimated_deaths"] = df_expanded["estimated_deaths"] / 12
df_expanded["estimated_death_rate"] = df_expanded["estimated_deaths"] / df_expanded["population"]
births = df_expanded.pivot(index="date", columns="cluster", values="estimated_births").to_numpy().astype(int) # (n_months, n_clusters)
death_rate = df_expanded.pivot(index="date", columns="cluster", values="estimated_death_rate").to_numpy() # (n_months, n_clusters)
# Initial demography
demo = demo[demo['year'] == start_year]['population'].values

# --- Indices ---
cluster_idx = df["cluster"].to_numpy().astype(int)
month_idx = df["month_idx"].to_numpy().astype(int)
year_idx = df["year_idx"].to_numpy().astype(int)

# --- Lengths ---
n_clusters = int(len(df['cluster'].unique()))
n_months = int(len(df["month_idx"].unique()))
n_years = int(df["year_idx"].max() + 1)
n_serotypes = len(sero_cols)

#########################
## Preparing the model ##
#########################

def critical_rho1(p, gamma):
    """Compute the coefficient of the first lag so that the sum of `p` AR coefficients: rho_k = 1/k**gamma sum to zero; resulting in a non-stationary process"""
    return 1 / pt.sum(1 / np.arange(1, p + 1)[None,:]**gamma, axis=1)

###############################
## Bayesian imputation model ##
###############################

from pymc.pytensorf import collect_default_updates

with pm.Model() as model:

    # --- Subtype Composition Model ---
    # p_{i,s,t} ~ Softmax(F_{i,s,t})

    ## Parameters 
    # Fitness sensitivity
    gamma = pt.as_tensor_variable(np.array([0.11, 0.1, 0.1, 0.1])) #pm.LogNormal("gamma", mu=-3, sigma=1, shape=n_serotypes)     # Intrinsic fitness 
    sigma_F = pm.HalfNormal("sigma_F", sigma=0.001)                       # Fitness innovation noise
    kappa = pt.as_tensor_variable(np.array([0.02, 0.02, 0.01, 0.005])) # pm.Beta("kappa", alpha=3, beta=100, shape=n_serotypes)       # Reported fractions 
    f_S0 = pt.as_tensor_variable(np.array([0.2, 0.6, 0.9, 0.95])) # pm.Beta("f_S0", alpha=2, beta=1, shape=n_serotypes)          # Fraction of total population susceptible to each serotype

    ## Initial states
    S0 = pm.Deterministic("S0", demo[None, :] * f_S0[:, None]).dimshuffle(1,0)  # susceptibles (n_clusters, n_serotypes)
    #p0 = pm.Dirichlet("p0", a=np.ones(n_serotypes), shape=(n_clusters, n_serotypes))

    p0 = pt.as_tensor_variable(np.tile(np.array([0.9, 0.001, 0.098, 0.001]), (n_clusters, 1)))


    ## Fitness innovations
    eps_F = pm.Normal("eps_F", mu=0.0, sigma=sigma_F, shape=(n_months-1, n_clusters, n_serotypes))

    # -------------------------------------------------
    # equations go here
    # -------------------------------------------------

    # https://www.youtube.com/watch?v=G9VWXZdbtKQ
    # https://pytensor.readthedocs.io/en/latest/library/scan.html 
    # https://gist.github.com/ricardoV94/a49b2cc1cf0f32a5f6dc31d6856ccb63#file-pymc_timeseries_ma-ipynb
    # https://becarioprecario.bitbucket.io/spde-gitbook/ch-intro.html

    def step(births_t, deaths_t, D_t, eps_t, S_prev, p_prev, gamma, kappa):

        # -----------------------
        # 1. Susceptible dynamics
        # -----------------------
        S_t = (
            S_prev
            - D_t[:, None] * p_prev / kappa[None, :]
            + births_t[:, None]
            - deaths_t[:, None] * S_prev
        )

        # soft lower bound to keep log well-defined
        S_t = 1 + pt.softplus(S_t - 1)

        # -----------------------
        # 2. Fitness from resource
        # -----------------------
        f_t = gamma[None, :] * pt.log(S_t)

        # mean fitness φ_t
        phi_t = pt.sum(p_prev * f_t, axis=1, keepdims=True)

        # ----------------------------
        # 3. Replicator update (Euler)
        # ----------------------------
        dp = p_prev * (f_t - phi_t) #+ eps_t
        p_t = p_prev + dp

        # numerical safety: keep on simplex
        p_t = pt.maximum(p_t, 0)
        p_t = p_t / pt.sum(p_t, axis=1, keepdims=True)

        return S_t, p_t

    # run step sequence
    (S_seq, p_seq), _ = pytensor.scan(
        fn=step,
        sequences=[
            pt.as_tensor_variable(births),
            pt.as_tensor_variable(death_rate),
            pt.as_tensor_variable(DENV_total),
            eps_F,
        ],
        outputs_info=[S0, p0],
        non_sequences=[gamma, kappa],
    )

    # attach initial states
    S = pm.Deterministic("S", pt.concatenate([S0[None, :, :], S_seq], axis=0))
    p = pm.Deterministic("p", pt.concatenate([p0[None, :, :], p_seq], axis=0))

    # Overdispersion models
    ## Time-independent hierarchical overdispersion (per cluster)
    d_cluster_hierarch = pm.HalfNormal("d_cluster_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
    d_cluster = pm.HalfNormal("d_cluster", sigma=d_cluster_hierarch, shape=n_clusters)
    phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_cluster, 1e-12))[None, :], n_months, axis=0))
    alpha = phi[:, :, None] * p # Broadcast phi over serotypes
    VIF = pm.Deterministic("VIF", (N_typed + phi) / (1 + phi)) # variance inflation of dirichlet multinomial compared to multinomial

    # --- Observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial)
    

    fig,ax=plt.subplots(nrows=6)
    cluster = 0
    # serotype distributions
    ax[0].plot(p.eval()[:,cluster,0], color='black', label='DENV-1')
    ax[0].plot(p.eval()[:,cluster,1], color='red', label='DENV-2')
    ax[0].plot(p.eval()[:,cluster,2], color='green', label='DENV-3')
    ax[0].plot(p.eval()[:,cluster,3], color='blue', label='DENV-4')
    ax[0].legend()
    # susceptibles
    ax[1].plot(S.eval()[:,cluster,0], color='black', label='DENV-1')
    ax[1].plot(S.eval()[:,cluster,1], color='red', label='DENV-2')
    ax[1].plot(S.eval()[:,cluster,2], color='green', label='DENV-3')
    ax[1].plot(S.eval()[:,cluster,3], color='blue', label='DENV-4')
    ax[1].legend()
    # DENV-1 counts observed versus modeled
    ax[2].plot(Y_obs.eval()[:, cluster, 0], color='black', linestyle='--', label='DENV-1')
    ax[2].plot(Y_multinomial[:, cluster, 0], color='black', label='DENV-1')
    # DENV-2 counts observed versus modeled
    ax[3].plot(Y_obs.eval()[:, cluster, 1], color='red', label='DENV-2')
    ax[3].plot(Y_multinomial[:, cluster, 1], color='black', label='DENV-2')
    # DENV-3 counts observed versus modeled
    ax[4].plot(Y_obs.eval()[:, cluster, 2], color='green', label='DENV-3')
    ax[4].plot(Y_multinomial[:, cluster, 2], color='black', label='DENV-3')
    # DENV-4 counts observed versus modeled
    ax[5].plot(Y_obs.eval()[:, cluster, 3], color='blue', label='DENV-4')
    ax[5].plot(Y_multinomial[:, cluster, 3], color='black', label='DENV-4')
    plt.show()
    plt.close()

#######################
## Running the model ##
#######################


# NUTS
draws=25
with model:
    trace = pm.sample(draws, tune=25, target_accept=0.99, chains=chains, cores=chains, init='adapt_diag', progressbar=True, idata_kwargs={'log_likelihood':True})


#######################
## Running the model ##
#######################

# Plot posterior predictive checks
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)
arviz.plot_ppc(posterior_predictive)
os.makedirs(f'{output_folder}/fig/posterior_predictive', exist_ok=True)
plt.savefig(f'{output_folder}/fig/posterior_predictive/ppc.pdf')
plt.close()    

# Expand data & take 2x as much samples
expanded_idata = trace.copy()
expanded_idata.posterior = trace.posterior.expand_dims(pred_id=2)
with model:
    posterior_predictive = pm.sample_posterior_predictive(
        expanded_idata,
        sample_dims=["chain", "draw", "pred_id"],
        extend_inferencedata=True,
    )

# Assume `trace` is the result of pm.sample()
arviz.to_netcdf(trace, f"{output_folder}/trace.nc")
arviz.to_netcdf(posterior_predictive, f"{output_folder}/posterior_predictive.nc")

# Traceplot
variables2plot = [
                'gamma', 'sigma_F', 'kappa', 'f_S0',
                ]

# Save traces
os.makedirs(f'{output_folder}/fig/trace', exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'{output_folder}/fig/trace/trace-{var}_typing-effort-model.pdf')
    plt.close()

###############################
## Diagnostics of dispersion ##
###############################


# ---- extract posterior predictive mean overdispersion proportions ----
# ---- phi summaries ----
## Overall
phi_gmean = np.exp(np.nanmean(np.log(trace.posterior["phi"].values)))
## Worst cluster
phi_gmean_worst = min(np.exp(np.nanmean(np.log(trace.posterior["phi"].values), axis=(0,1,2))))

# ---- VIF summaries ----
## Overall
VIF_gmean = np.exp(np.nanmean(np.log(trace.posterior["VIF"].values)))
## Worst cluster
VIF_gmean_worst = max(np.exp(np.nanmean(np.log(trace.posterior["VIF"].values), axis=(0,1,2))))

# ---- print summary to a log file ----
with open(os.path.join(output_folder, "model_quality.log"), "w") as f, redirect_stdout(f):
    print("\phi geometric mean across clusters:", phi_gmean)
    print("\phi geometric mean of the worst cluster:", phi_gmean_worst)
    print("VIF geometric mean across clusters:", VIF_gmean)
    print("VIF geometric mean of the worst cluster:", VIF_gmean_worst)



