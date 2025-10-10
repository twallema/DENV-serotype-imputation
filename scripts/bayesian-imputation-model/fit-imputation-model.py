import os
import arviz
import argparse
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
parser.add_argument("-region_filename", type=str, help="Spatial aggregation clustering was performed on.", default='rgint')
parser.add_argument("-chains", type=int, help="Number of parallel chains.", default=4)
parser.add_argument("-ID", type=str, help="Sampler output name.")
parser.add_argument("-p", type=int, help="Order of AR(p) process.", default=1)
parser.add_argument("-distance_matrix", type=str_to_bool, help="Use distance matrix versus adjacency matrix.", default=False)
args = parser.parse_args()

# assign to desired variables
region_filename = args.region_filename
chains = args.chains
ID = args.ID
p = args.p
distance_matrix = args.distance_matrix

# Make folder structure
output_folder=f'../../data/interim/bayesian-imputation-model_output/AR({p})/distance_matrix-{distance_matrix}/{ID}_{datetime.today().strftime("%Y-%m-%d")}' # Path to backend
# check if samples folder exists, if not, make it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

########################
## Preparing the data ##
########################

# Load clusters
# >>>>>>>>>>>>>

clusters = pd.read_csv(f'../../data/interim/clusters/clusters_{region_filename}.csv')
region = clusters.columns.to_list()[0]

# Load mapping
# >>>>>>>>>>>>

mapping = pd.read_csv(f'../../data/interim/spatial_units_mapping.csv')

# Distance matrix
# ~~~~~~~~~~~~~~~

if distance_matrix == False:
    # Load adjacency matrix
    D = pd.read_csv(f'../../data/interim/clusters/adjacency_matrix_{region_filename}.csv', index_col=0).values
else:
    # Load distance matrix
    D = pd.read_csv(f'../../data/interim/clusters/distance_matrix_{region_filename}.csv', index_col=0).values


# Incidence data
# ~~~~~~~~~~~~~~

# Fetch incidence data
df = pd.read_csv('../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv', parse_dates=['date'])

# 1. Check if all columns are present
sero_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4"]
required_cols = ["CD_MUN", "date", "DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]
assert all(col in df.columns for col in required_cols)

# 2. Sort for safety
df = df.sort_values(["CD_MUN", "date"]).reset_index(drop=True)

# 3. Take only from startdate
df = df[df['date'] > startdate]

# 4. Aggregate to the spatial clusters
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

# 5. Add number of serotyped cases
df["N_typed"] = df[sero_cols].sum(axis=1, skipna=False)           # if serotypes available --> sum them
df.loc[df[sero_cols].isna().all(axis=1), 'N_typed'] = np.nan      # if all serotypes are Nan --> N_typed = 0 --> Wait, I don't think this is appropriate.

# 6. Compute delta (typing fraction)
df["delta"] = df["N_typed"] / df["DENV_total"]
df['delta'] = df['delta'].where(df['N_typed'] > 0, np.nan) # When N_typed == 0, we don't know delta — mark as missing
df["delta"] = df["delta"].clip(lower=1e-12, upper=1 - 1e-12)

# 7. Compute year and month index
df["year"] = pd.to_datetime(df["date"]).dt.year
df["year_idx"] = df["year"] - df["year"].min()
df['month_idx'], _ = pd.factorize(df['date'])


# 8. Build PyMC arrays
# --- For Beta model (typing fraction, always available) ---
delta_obs = df["delta"].to_numpy().astype(float)
N_total = df["DENV_total"].to_numpy().astype(int)
# --- For Multinomial model (subtypes, only when typed) ---
Y_multinomial = df[sero_cols].to_numpy().astype(int)
N_typed = df["N_typed"].to_numpy().astype(int)
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
    """Compute the coefficient of the first lag so that the sum of p AR coefficients: rho_k = 1/k**gamma sum to zero; resulting in a non-stationary process"""
    return 1 / pt.sum(1 / np.arange(1, p + 1)[None,:]**gamma[:,None], axis=1)

###############################
## Bayesian imputation model ##
###############################

with pm.Model() as model:

    # --- Subtype Composition Model ---
    # p_{i,s,t} ~ Softmax(\theta_{i,s,t})
    # \theta_{i,s,t} = \sum_{k=1}^p \rho_k \alpha_{i,s,t-k} +  \kappa_{i,s,t}^{corr} + \kappa_{i,s,t}^{uncorr}          # AR(p) process
    # \kappa_{i,s,t}^{corr} ~ Normal(0, f_{corr} * \sigma^2  * chol(Q))                                                 # spatially correlated noise
    # \kappa{i,s,t}^{uncorr} ~ Normal(0, (1-f_{corr}) * \sigma^2)                                                       # spatially uncorrelated noise


    # Try to combine an AR(p) with a CAR prior on every timestep in the past
    ## Regularisation of the overall noise & split between spatially structured and unstructured noise
    total_sigma = pm.HalfNormal("total_sigma", sigma=0.1)
    proportion_uncorr = pm.Beta("proportion_uncorr", alpha=1, beta=25)  # proportion of noise that is unstructured (encourages structured noise)
    uncorr_sigma = pm.Deterministic("uncorr_sigma", proportion_uncorr * total_sigma) * pt.ones(n_serotypes)
    corr_sigma = pm.Deterministic("corr_sigma", (1 - proportion_uncorr) * total_sigma) * pt.ones(n_serotypes)

    ## Temporal correlation structure: Harmonically decaying weights summing to one (guarantees non-stationarity).
    gamma = pt.ones(n_serotypes)
    first_lag = pm.Deterministic("first_lag", critical_rho1(p,gamma))
    rho = pm.Deterministic("rho", first_lag[:,None] / ((np.arange(1, p + 1)[None,:])**gamma[:,None]))
    AR_coefficients_sum = pm.Deterministic("AR_coefficients_sum", pt.sum(rho, axis=1))

    ## Priors for spatial correlation radius (zeta)
    if distance_matrix: 
        zeta = pm.HalfNormal("zeta", sigma=100)
    else:
        zeta = -1
        pass

    ## Priors for spatial correlation strength (a)
    a_car = 1

    # Pair-wise kernel first
    # D_shared: (n_clusters, n_clusters)
    # zeta_car: (n_serotypes, p)
    # We need to broadcast D_shared against zeta
    W = pt.exp(-D[None, :, :] / zeta)
    # Construct degree tensor (matrix equivalent: row sums of weighted distance matrix on diagonal of eye(n_clusters))
    degree = pt.sum(W, axis=-1)[:, :, None]
    I = pt.eye(n_clusters)[None, :, :]
    D = I * degree
    # Q = D - a * W + jitter
    jitter = 1e-6 * pt.diag(pt.ones(n_clusters))
    jitter = jitter[None, :, :]
    Q = D - a_car * W + jitter

    # Compute the Cholesky of Q
    chol = pt.slinalg.cholesky(Q)

    # Scale with the noise
    chol = chol * corr_sigma[:, None, None]  # (n_serotypes, n_clusters, n_clusters)

    # Initialise AR(p) initial condition
    AR_init = pm.Normal("AR_init", mu=0, sigma=1, shape=(p, n_serotypes, n_clusters))

    # Initialise spatial innovation noise (one per lag)
    #epsilon_corr = pm.Normal("epsilon_corr", 0, 1, shape=(n_months - p, n_serotypes, n_clusters))

    # --- temporally-smoothed spatial innovations (replace epsilon_corr) ---
    # We construct a GaussianRandomWalk for each (serotype, cluster) entity,
    # then reshape it into the time-first ordering your arp_step expects.

    # sigma for the per-entity random walks (choose sensible scale; chol already scales spatial amplitude)
    sigma_eps = pm.HalfNormal("sigma_eps", sigma=1.0)  # tune as needed

    # init distribution for each entity (non-centered style is handled by GaussianRandomWalk internally)
    eps_init = pm.Normal.dist(mu=0.0, sigma=1.0, shape=(n_serotypes * n_clusters,))

    # eps_rw shape: (n_entities (= n_serotypes*n_clusters), n_months - p)
    eps_rw = pm.GaussianRandomWalk(
        "eps_rw",
        sigma=sigma_eps,
        init_dist=eps_init,
        shape=(n_serotypes * n_clusters, n_months - p),
    )

    # reshape to (n_serotypes, n_clusters, n_months - p) and then dimshuffle to (n_months - p, n_serotypes, n_clusters)
    epsilon_corr = eps_rw.reshape((n_serotypes, n_clusters, n_months - p)).dimshuffle(2, 0, 1)
    # epsilon_corr now has the same shape as your original (n_months - p, n_serotypes, n_clusters)

    # Initialise random noise
    epsilon_uncorr = pm.Normal("epsilon_uncorr", mu=0, sigma=1, shape=(n_months - p, n_serotypes, n_clusters))


    def arp_step(epsilon_corr_t, epsilon_uncorr_t, previous_vals, rho, chol, uncorr_sigma):
        """
        previous_vals: (p, n_serotypes, n_clusters)
        epsilon_t: (n_serotypes, n_clusters)
        epsilon_uncorr_t: (n_serotypes, n_clusters)
        """

        spatial_noise = pt.batched_dot(epsilon_corr_t, chol)
        AR_noise = epsilon_uncorr_t * uncorr_sigma[:, None]
        AR_mean = []
        for lag in range(p):
            # Apply temporal weight rho_k (serotype-specific)
            AR_mean.append(rho[:, lag][:, None] * previous_vals[lag])

        # Sum weighted AR and spatial noise over lags
        new_vals = sum(AR_mean) + spatial_noise + AR_noise  # (n_serotypes, n_clusters)

        # Shift lag window: insert new_vals at position 0
        updated_vals = pt.concatenate(
            [new_vals[None, :, :], previous_vals[:-1]], axis=0
        )  # (p, n_serotypes, n_clusters)

        return updated_vals
    
    sequences, _ = pytensor.scan(
        fn=arp_step,
        sequences=[epsilon_corr, epsilon_uncorr],
        outputs_info=AR_init,
        non_sequences=[rho, chol, uncorr_sigma],
    )


    # sequences: (n_months - p, p, n_serotypes, n_clusters)
    # AR_init: (p, n_serotypes, n_clusters)
    theta_log_final = pt.concatenate([pt.repeat(AR_init[None, :, :, :], p, axis=0), sequences], axis=0)
    # Step 3: slice lag zero (p=0) over full time axis
    theta_log_final = theta_log_final[:, 0, :, :]  # shape (n_months, n_serotypes, n_clusters)
    # Step 4: convert to flat format
    theta_log = theta_log_final.reshape((len(df), n_serotypes))


    # OPTION 1: DirichletMultinomial >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Softmax transform AR+CAR model
    p = pm.Deterministic("p", pm.math.softmax(theta_log, axis=1))

    # Hierarchical prior for RW step size per cluster
    mu_sigma_rw = pm.HalfNormal("mu_sigma_rw", sigma=1)
    sigma_rw_cluster = pm.HalfNormal("sigma_rw_cluster", sigma=mu_sigma_rw, shape=n_clusters)

    # Cluster-specific Gaussian random walks for log(phi)
    logphi_rw = pm.GaussianRandomWalk(
        "logphi_rw",
        sigma=pt.transpose(pt.repeat(sigma_rw_cluster[:, None], n_months-1, axis=1)),
        init_dist=pm.Normal.dist(mu=5, sigma=1, shape=n_clusters),
        shape=(n_clusters, n_months),
    )
    phi_obs = pm.Deterministic("phi_obs", pm.math.exp(logphi_rw.flatten()))

    # Compute Dirichlet concentration parameter
    alpha = phi_obs[:, None] * p   # p already computed from theta_log

    # --- Observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial)


#######################
## Running the model ##
#######################

# NUTS
with model:
    trace = pm.sample(50, tune=50, target_accept=0.99, chains=chains, cores=chains, init='adapt_diag', progressbar=True, idata_kwargs={'log_likelihood':True})


#######################
## Running the model ##
#######################

# Plot posterior predictive checks
with model:
    ppc = pm.sample_posterior_predictive(trace)
arviz.plot_ppc(ppc)
plt.savefig(f'{output_folder}/ppc.pdf')
plt.close()    

# Expand data & take 10x as much samples
expanded_idata = trace.copy()
expanded_idata.posterior = trace.posterior.expand_dims(pred_id=5)
with model:
    ppc = pm.sample_posterior_predictive(
        expanded_idata,
        sample_dims=["chain", "draw", "pred_id"],
        extend_inferencedata=True,
    )

# Assume `trace` is the result of pm.sample()
arviz.to_netcdf(trace, f"{output_folder}/trace.nc")
arviz.to_netcdf(ppc, f"{output_folder}/ppc.nc")

# Traceplot
variables2plot = [
                    'total_sigma', 'proportion_uncorr', 'corr_sigma', 'uncorr_sigma', 'mu_sigma_rw', 'sigma_rw_cluster',
                ]
if distance_matrix:
    variables2plot += ['zeta',]


for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'{output_folder}/trace-{var}_typing-effort-model.pdf')
    plt.close()

# Print summary
summary_df = arviz.summary(trace, round_to=3)
print(summary_df)


###############################
## Diagnostics of dispersion ##
###############################


import numpy as np
import arviz as az
from scipy.stats import gmean

# ---- helpers ----
def clr(proportions, pseudocount=1e-8):
    X = np.asarray(proportions) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=1, keepdims=True)
    return logX - gm

def aitchison_dist_rows(P, Q):
    C1 = clr(P)
    C2 = clr(Q)
    return np.linalg.norm(C1 - C2, axis=1)

# ---- extract posterior predictive mean proportions ----
# ppc.posterior_predictive["Y_obs"] shape: (chains, draws, n_obs, K) OR (samples, n_obs, K)
pp = ppc.posterior_predictive["Y_obs"].values
pp = pp.reshape(-1, pp.shape[-2], pp.shape[-1])  # (n_pp_samples, n_obs, K)
pp_props = pp / pp.sum(axis=2, keepdims=True)    # proportions per pp draw
pp_mean_props = pp_props.mean(axis=0)            # (n_obs, K) posterior predictive mean proportions

# observed proportions
Y = Y_multinomial  # shape (n_obs, K)
obs_props = Y / Y.sum(axis=1, keepdims=True)
valid_idx = Y.sum(axis=1) > 0
obs_props_valid = obs_props[valid_idx]
pp_mean_props_valid = pp_mean_props[valid_idx]
d_obs_to_mean = aitchison_dist_rows(obs_props_valid, pp_mean_props_valid)
RMSAD = np.sqrt(np.mean(d_obs_to_mean**2))
MSAD = np.mean(d_obs_to_mean**2)

# ---- phi summaries ----
phi_samples = trace.posterior["phi_obs"].values.flatten()  # (chains, draws, n_clusters)
phi_median = np.median(phi_samples)
phi_geo_mean = gmean(phi_samples)
phi_hdi = az.hdi(phi_samples, hdi_prob=0.95)  # array (n_clusters, 2)

# ---- print summary ----
print("RMS Aitchison distance (global):", RMSAD)
print("MSAD:", MSAD)
print("phi median across clusters:", phi_median)
print("phi geometric mean across clusters:", phi_geo_mean)




