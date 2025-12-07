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
startdate = datetime(1900,1,1)

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
parser.add_argument("-p", type=int, help="Order of AR(p) process.", default=1)
parser.add_argument("-q", type=int, help="Order of MA(q) process.", default=1)
args = parser.parse_args()

# assign to desired variables
spatial_aggregation = args.spatial_aggregation
chains = args.chains
ID = args.ID
p = args.p
q = args.q

# pipeline output folder
abs_dir = os.path.dirname(__file__) # make sure all referenced paths are relative to the lcoation of this file and not the terminal's pwd
output_folder = os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/bayesian-imputation-model_output/ARMA({p},{q})/')
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

# 3. Take only from startdate
df = df[df['date'] > startdate]

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
# --- For Multinomial model (subtypes, only when typed) ---
# Total number of typed cases
N_typed = df.pivot(index="date", columns="cluster", values="N_typed").to_numpy().astype(int)    # (n_months, n_clusters)
# Number of cases per DENV serotype
Y_list = []
for col in sero_cols:
    Y_mat = df.pivot(index="date", columns="cluster", values=col).to_numpy()
    Y_list.append(Y_mat)
Y_multinomial = np.stack(Y_list, axis=2).astype(int)    # (n_months, n_clusters, n_serotypes)
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
    # p_{i,s,t} ~ Softmax(log \theta_{i,s,t})
    # log \theta_{i,s,t} = \sum_{k=1}^p \rho_k \theta_{i,s,t-k} + \sum_{k=1}^q \psi_k \kappa_{i,s,t-k} + \kappa_{i,s,t}^{corr}      # ARMA(p,q) with                              # AR(p) process with RW(1) CAR innovation noise
    # \kappa_{i,s,t}^{corr} ~ Normal(0, Q^{-1})                                                                                     # spatially correlated noise

    ## Regularisation of the overall noise
    total_sigma = pm.HalfNormal("total_sigma", sigma=0.003)
    a_CAR = pm.Beta("a_CAR", alpha=2, beta=1)
    a_CAR_trunc = pm.Deterministic("a_CAR_trunc", a_CAR*0.99)

    ## Temporal correlation structure: Quadratically decaying weights (gamma=2) summing to one to guarantee non-stationarity
    gamma = 2
    first_lag_p = pm.Deterministic("first_lag_p", critical_rho1(p, gamma))
    first_lag_q = pm.Deterministic("first_lag_q", critical_rho1(q, gamma))
    rho = pm.Deterministic("rho", first_lag_p / ((np.arange(1, p + 1))**gamma))
    psi = pm.Deterministic("psi", first_lag_q / ((np.arange(1, q + 1))**gamma))

    ## Priors for spatial correlation
    W = pt.constant(W)                                  # fixed adjacency matrix in graph
    D = pt.diag(pt.sum(W, axis=1))                      # degree matrix
    jitter = 1e-6 * pt.eye(n_clusters)                  # small jitter to stabilize computation
    Q = D - a_CAR_trunc * W + jitter                    # build precision matrix Q = D - a * W + jitter  (shape: n_clusters x n_clusters)
    L_Q = pt.slinalg.cholesky(Q)                        # Cholesky of precision: Q = L_Q @ L_Q.T  (lower-triangular)
    L_cov = pt.slinalg.solve(L_Q, pt.eye(n_clusters))   # Compute L_cov = L_Q^{-1} such that Sigma = L_cov @ L_cov.T = Q^{-1}
    chol_cov_scaled = total_sigma * L_cov               # scale covariance cholesky by total_sigma (put scale inside chol)

    ## Initialise AR(p) initial condition
    AR_init = pm.Normal("AR_init", mu=0, sigma=1, shape=(p, n_clusters, n_serotypes))
    kappa_init = pm.Normal("kappa_init", mu=0, sigma=1, shape=(q, n_clusters, n_serotypes))

    # -------------------------------------------------
    # log_phi = ARMA(p,q) + \epsilon_t
    # \epsilon_t ~ MvNormal(0, total_sigma**2 * Q^{-1})
    # -------------------------------------------------

    # https://www.youtube.com/watch?v=G9VWXZdbtKQ
    # https://pytensor.readthedocs.io/en/latest/library/scan.html 
    # https://gist.github.com/ricardoV94/a49b2cc1cf0f32a5f6dc31d6856ccb63#file-pymc_timeseries_ma-ipynb
    # https://becarioprecario.bitbucket.io/spde-gitbook/ch-intro.html

    # ---- Step 1: Generate a sequence of spatially correlated innovations ----
    kappa_sequence = pm.MvNormal(
        "kappa_sequence",
        mu=0,
        chol=chol_cov_scaled,
        shape=(n_months, n_serotypes, n_clusters) # last axis has to be Mv axis
    )
    kappa_sequence = kappa_sequence.dimshuffle(0, 2, 1) # (n_months, n_clusters, n_serotypes)

    # ---- Step 2: Deterministically reconstruct the ARMA(p,q) sequence using the sequence of innovations ----
    def reconstruct_arma_pq(kappa_init, AR_init, kappa_sequence, rho, psi):
        """
        Reconstruct the ARMA(p,q) sequence deterministically from innovations
        """

        # define update function
        def step(kappa_t, prev_kappa, prev_vals, rho, psi):
            # update the states
            ARp = pt.tensordot(rho, prev_vals, axes=[0,0])
            MAq = pt.tensordot(psi, prev_kappa, axes=[0,0])
            new_vals = ARp + MAq + kappa_t
            # Shift lag windows
            new_kappa = pt.concatenate([kappa_t[None, :, :], prev_kappa[:-1]], axis=0)
            new_vals = pt.concatenate([new_vals[None, :, :], prev_vals[:-1]], axis=0)
            return new_kappa, new_vals
        
        # perform scanning
        [_, sequence_vals], _ = pytensor.scan(
            fn=step,
            sequences=kappa_sequence,
            outputs_info=[kappa_init, AR_init],
            non_sequences=[rho, psi],
        )
        return sequence_vals[:, 0, :, :] # return states

    # Wrap into a pm.Deterministic variable
    theta_log = pm.Deterministic(
        "theta_log",
        reconstruct_arma_pq(kappa_init, AR_init, kappa_sequence, rho, psi)
    )   # (n_months, n_clusters, n_serotypes)

    # -------------------
    # p = softmax(logphi)
    # -------------------

    p = pm.Deterministic("p", pm.math.softmax(theta_log, axis=2))

    # Overdispersion models
    ## Time-independent hierarchical overdispersion (per cluster)
    d_cluster_hierarch = pm.HalfNormal("d_cluster_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
    d_cluster = pm.HalfNormal("d_cluster", sigma=d_cluster_hierarch, shape=n_clusters)
    phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_cluster, 1e-12))[None, :], n_months, axis=0))
    alpha = phi[:, :, None] * p # Broadcast phi over serotypes
    VIF = pm.Deterministic("VIF", (N_typed + phi) / (1 + phi)) # variance inflation of dirichlet multinomial compared to multinomial

    # --- Observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial)

#######################
## Running the model ##
#######################


# NUTS
draws=500
with model:
    trace = pm.sample(draws, tune=1500, target_accept=0.99, chains=chains, cores=chains, init='adapt_diag', progressbar=True, idata_kwargs={'log_likelihood':True})


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
                'total_sigma', 'a_CAR_trunc', 'd_cluster_hierarch', 'd_cluster',
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



