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
parser.add_argument("-chains", type=int, help="Number of parallel chains.", default=1)
parser.add_argument("-ID", type=str, help="Sampler output name.")
args = parser.parse_args()

# assign to desired variables
region_filename = args.region_filename
chains = args.chains
ID = args.ID

# Make folder structure
output_folder=f'../../data/interim/bayesian-imputation-model_output/{ID}_{datetime.today().strftime("%Y-%m-%d")}' # Path to backend
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
    """Compute the coefficient of the first lag so that the sum of `p` AR coefficients: rho_k = 1/k**gamma sum to zero; resulting in a non-stationary process"""
    return 1 / pt.sum(1 / np.arange(1, p + 1)[None,:]**gamma[:,None], axis=1)

###############################
## Bayesian imputation model ##
###############################

# ---- Helper kernel functions ----
def matern32(x, ell):
    """Matern 3/2 kernel."""
    sqrt3 = np.sqrt(3.0)
    return (1 + sqrt3 * x / ell) * pt.exp(-sqrt3 * x / ell)

def ou_kernel(x, ell):
    """Ornstein-Uhlenbeck (Exponential) kernel."""
    return pt.exp(-x / ell)

with pm.Model() as model:

    # --- Kernel hyperpriors ---
    total_sigma = pm.HalfNormal("total_sigma", sigma=0.02) # --> Final inferred value of around ~0.3-0.35 yields optimal smoothing
    ell_s = pm.HalfNormal("ell_s", sigma=100)
    ell_t = pm.HalfNormal("ell_t", sigma=12)

    # --- Spatial covariance ---
    K_space = matern32(pt.as_tensor_variable(D), ell_s) + 1e-6 * pt.eye(n_clusters)
    chol_space = pt.slinalg.cholesky(K_space)  # (n_clusters x n_clusters)

    # --- Temporal covariance ---
    t_idx = np.arange(n_months)
    Tdiff = np.abs(t_idx[:, None] - t_idx[None, :])
    Tdiff = pt.as_tensor_variable(Tdiff)  # shape (n_months, n_months)
    K_time = ou_kernel(Tdiff, ell_t) + 1e-6 * pt.eye(n_months)
    chol_time = pt.slinalg.cholesky(K_time)  # (n_months x n_months)

    # --- Standard normal latent field ---
    Z = pm.Normal("Z", mu=0, sigma=total_sigma, shape=(n_clusters, n_months, n_serotypes))

    # --- Apply Kronecker Cholesky ---
    # Using vectorized batched_dot: Ls @ Z[:,:,s] @ Lt.T for each serotype
    theta_list = []
    for s in range(n_serotypes):
        theta_s = pt.dot(chol_space, Z[:,:,s]).dot(chol_time.T)
        theta_list.append(theta_s)
    theta = pt.stack(theta_list, axis=-1)  # (n_clusters, n_months, n_serotypes)
    theta = theta.dimshuffle(1,0,2)        # (n_months, n_clusters, n_serotypes)

    # Flatten to match observation ordering: (n_months*n_clusters, n_serotypes)
    theta_for_softmax = theta.reshape((n_months * n_clusters, n_serotypes))
    p = pm.Deterministic("p", pm.math.softmax(theta_for_softmax, axis=1))

    # --- Overdispersion / phi ---
    logphi_rw_sigma_hierarchical_mean = pm.HalfNormal("logphi_rw_sigma_hierarchical_mean", sigma=1)
    logphi_rw_sigma_cluster = pm.HalfNormal("logphi_rw_sigma_cluster", sigma=logphi_rw_sigma_hierarchical_mean, shape=n_clusters)
    logphi_rw = pm.GaussianRandomWalk(
        "logphi_rw",
        sigma=pt.transpose(pt.repeat(logphi_rw_sigma_cluster[:, None], n_months-1, axis=1)),
        init_dist=pm.Normal.dist(mu=5, sigma=1, shape=n_clusters),
        shape=(n_clusters, n_months),
    )

    # Flatten phi to match ordering: (n_months*n_clusters,)
    phi_obs_t = pm.Deterministic("phi_obs_t", pm.math.exp(logphi_rw.T).flatten())

    # --- Dirichlet-Multinomial likelihood ---
    alpha = phi_obs_t[:, None] * p
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial)

    # Compute variance inflation of dirichlet multinomial compared to multinomial
    VIF = pm.Deterministic("VIF", (N_typed + phi_obs_t) / (1 + phi_obs_t))

#######################
## Running the model ##
#######################

# NUTS
draws=50
with model:
    trace = pm.sample(draws, tune=50, target_accept=0.99, chains=chains, cores=chains, init='adapt_diag', progressbar=True, idata_kwargs={'log_likelihood':True})


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
expanded_idata.posterior = trace.posterior.expand_dims(pred_id=10)
with model:
    ppc = pm.sample_posterior_predictive(
        expanded_idata,
        sample_dims=["chain", "draw", "pred_id"],
        extend_inferencedata=True,
    )

# Assume `trace` is the result of pm.sample()
arviz.to_netcdf(trace, f"{output_folder}/trace.nc")
arviz.to_netcdf(ppc, f"{output_folder}/ppc.nc")

# Traceplots
variables2plot = [
                   'total_sigma', 'ell_s', 'ell_t', 'logphi_rw_sigma_hierarchical_mean', 'logphi_rw_sigma_cluster',
                ]

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


# ---- extract posterior predictive mean overdispersion proportions ----
# ---- phi summaries ----
## Overall
x = trace.posterior["phi_obs_t"].values # (chains, draws, n_months * n_clusters)
x = x.reshape(chains, draws, n_months, n_clusters)
x = x.reshape(-1, n_months, n_clusters)
phi_gmean = np.exp(np.nanmean(np.log(x)))
## Worst cluster
phi_gmean_worst = min(np.exp(np.nanmean(np.log(x), axis=(0,1))))

# ---- VIF summaries ----
## Overall
x = trace.posterior["VIF"].values # (chains, draws, n_months * n_clusters)
x = x.reshape(chains, draws, n_months, n_clusters)
x = x.reshape(-1, n_months, n_clusters)
VIF_gmean = np.exp(np.nanmean(np.log(x)))
## Worst cluster
VIF_gmean_worst = max(np.exp(np.nanmean(np.log(x), axis=(0,1))))

# ---- print summary ----
print("\phi(t) geometric mean:", phi_gmean)
print("\phi(t) geometric mean of the worst cluster:", phi_gmean_worst)
print("VIF geometric mean:", VIF_gmean)
print("VIF geometric mean of the worst cluster:", VIF_gmean)



