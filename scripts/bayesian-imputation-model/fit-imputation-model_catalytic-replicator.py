import os
import ast
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
start_year = 1998
start_month = 9
end_year = 2024
assert start_year >= 1996, "earliest start_year is 1996."

# helper function for argument parsing
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# arguments determine the model + data combo used to forecast
# How to run: python fit-model.py -ID test -p 2 -distance_matrix False
parser = argparse.ArgumentParser()
parser.add_argument("-ID", type=str, help="Identifier of the pipeline run.")
parser.add_argument("-spatial_aggregation", type=str, help="Spatial aggregation clustering was performed on.")
parser.add_argument("-chains", type=int, help="Number of parallel chains.", default=4)
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

demo = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/IBGE_population-projections/pop-births-deaths_mun_1996-2024.csv'))
demo = demo.rename(columns={'geocode': 'CD_MUN'})
demo = demo.merge(mapping[['CD_MUN', 'cluster']], on='CD_MUN', how='left')
demo = demo.groupby(['cluster', 'year'], as_index=False)['population'].sum()

# Compute births and death rates per cluster
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

bd = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/IBGE_population-projections/pop-births-deaths_mun_1996-2024.csv'))
bd = bd.merge(mapping[['CD_MUN', 'cluster']], on='CD_MUN', how='left')
bd = bd.groupby(['year', 'cluster'], as_index=False).agg(estimated_births=('estimated_births', 'sum'),estimated_deaths=('estimated_deaths', 'sum'), population=('population', 'sum'))

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

# only do first X clusters
df = df[df['cluster'].isin([23, 24])]

# 3. Take only from start_year to end_year
df_alldates = df.copy()
df = df[((df['date'] > datetime(start_year,start_month,1)) & (df['date'] <= datetime(end_year,12,31)))]

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
demo = demo[((demo['year'] == start_year) & (demo['cluster'].isin([23, 24])))]['population'].values 

# --- Indices ---
cluster_idx = df["cluster"].to_numpy().astype(int)
month_idx = df["month_idx"].to_numpy().astype(int)
year_idx = df["year_idx"].to_numpy().astype(int)

# --- Lengths ---
n_clusters = int(len(df['cluster'].unique()))
n_months = int(len(df["month_idx"].unique()))
n_years = int(df["year_idx"].max() + 1)
n_serotypes = len(sero_cols)

# Estimate initial serotype distribution p0 from all data before startdate
# Use the mean of the posterior of Dirichlet-Multinomial model with symmetric prior alpha = 1/2 (Jeffrey's prior = uninformative prior)
d = df_alldates[df_alldates['date'] <= datetime(start_year, start_month, 1)].groupby(by='cluster')[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].sum()
#d = df[df['year'] == min(df['year'])].groupby(by='cluster')[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].sum()
cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4"]
alpha = 1/2
p0 = (d[cols] + alpha).div(d[cols].sum(axis=1) + len(cols) * alpha, axis=0)
# But we set DENV_4 initially to zero and renormalise --> comes through seeding
p0['DENV_3'] = 0
p0['DENV_4'] = 0
p0 = p0.div(p0.sum(axis=1), axis=0)

# Rolling-window version for seeding
window = 12
# Aggregate by month and cluster
df_monthly = df.groupby(['date', 'cluster'])[cols].sum().reset_index()
# Get sorted list of clusters and months
clusters = sorted(df_monthly['cluster'].unique())
months = sorted(df_monthly['date'].unique())
# Initialize output array
n_months = len(months)
n_clusters = len(clusters)
n_serotypes = len(cols)
rolling_array = np.zeros((n_months, n_clusters, n_serotypes))
# Compute rolling Bayesian for each cluster
for c_idx, cl in enumerate(clusters):
    df_cl = df_monthly[df_monthly['cluster'] == cl].set_index('date').reindex(months, fill_value=0)[cols]
    for i in range(n_months):
        start_idx = max(0, i - window + 1)
        window_sum = df_cl.iloc[start_idx:i+1].sum()
        bayes = (window_sum + alpha) / (window_sum.sum() + n_serotypes*alpha)
        rolling_array[i, c_idx, :] = bayes.values
estimated_proportions = rolling_array   # shape: (n_months, n_clusters, n_serotypes)

# Make a serotype introduction mask; shape: (n_months, n_serotypes)
unique_dates = np.sort(df["date"].unique())
years = pd.DatetimeIndex(unique_dates).year
intro_mask = np.ones((len(unique_dates), 4))
intro_mask[:, 3] = (years >= 2008).astype(int)
intro_mask[:, 2] = (years >= 1999).astype(int)

# Load in the transmission model state mappings
## susceptibles
S_mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/raw/transmission_model/susceptible_states.csv'))
list_columns = ["susc_to_heterol_inf", "susc_to_homol_inf"]
# Convert strings to actual lists
for col in list_columns:
    S_mapping[col] = S_mapping[col].apply(ast.literal_eval)
## infectious/protected
IP_mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/raw/transmission_model/infectious_protected_states.csv'))

# Generate the matrices used to update the system
## define lengths
n_I = len(IP_mapping)
n_S = len(S_mapping)
## I-update-indices
sero_idx = pt.constant(IP_mapping["currently_infected_with"].values - 1)
het_idx  = pt.constant(IP_mapping["S_to_I_heterol"].values)
hom_idx  = pt.constant(IP_mapping["S_to_I_homol"].values)
## births
birth_vec = np.zeros(n_S)
birth_vec[0] = 1.0
birth_vec_t = pt.constant(birth_vec)
## C --> Maps infected states onto serotypes
C = np.zeros((n_I, n_serotypes))
C[np.arange(n_I), sero_idx.eval()] = 1.0
C = pt.constant(C)
## W --> Maps protected states waning to S
W = pt.constant(np.eye(n_S)[IP_mapping["P_to_S"]])
## Susceptibility of S states to serotypes (M = H_het + f * H_hom)
H_het = np.zeros((n_S, n_serotypes))
H_hom = np.zeros((n_S, n_serotypes))
for j, row in S_mapping.iterrows():
    for k in row["susc_to_heterol_inf"]:
        H_het[j, int(k)-1] = 1.0
    for k in row["susc_to_homol_inf"]:
        H_hom[j, int(k)-1] = 1.0
H_het = pt.constant(H_het)
H_hom = pt.constant(H_hom)
## Flow of S states into I states (heterologous and homologous)
K_het = np.zeros((n_I, n_S))
K_hom = np.zeros((n_I, n_S))
for i, row in IP_mapping.iterrows():
    K_het[i, int(row["S_to_I_heterol"])] = 1.0
    K_hom[i, int(row["S_to_I_homol"])] = 1.0
K_het = pt.constant(K_het)
K_hom = pt.constant(K_hom)
## Map serotype-specific FOI to the per I state FOI
L = np.zeros((n_serotypes, n_I))
for i, s in enumerate(IP_mapping["currently_infected_with"].values - 1):
    L[s, i] = 1.0
L = pt.constant(L)

# post-hoc observation mapping
## Delta_I mapping to (cluster, degree, serotype)
O = np.zeros((n_I, IP_mapping["no_prior_heterol_inf"].max() + 1, n_serotypes))
for i, row in IP_mapping.iterrows():
    d = int(row["no_prior_heterol_inf"])
    s = int(row["currently_infected_with"]) - 1
    O[i, d, s] = 1.0
O = pt.constant(O)
## S (slots) mapping to (cluster, degree, serotype)
### make two maps: one to heterologous and one to homologous 
### susceptibility slots = heterologous + f_i * homologous
R_het = np.zeros((n_S, S_mapping["no_prior_heterol_inf"].max() + 1, n_serotypes))
R_hom = np.zeros((n_S, S_mapping["no_prior_heterol_inf"].max() + 1, n_serotypes))
for i, row in S_mapping.iterrows():
    d = int(row["no_prior_heterol_inf"])
    for s in row["susc_to_heterol_inf"]:
        R_het[i, d, s-1] = 1.0
    for s in row["susc_to_homol_inf"]:
        R_hom[i, d, s-1] = 1.0
R_het = pt.constant(R_het)
R_hom = pt.constant(R_hom)
## S (individuals) mapping to (cluster, degree)
D = np.zeros((n_S, S_mapping["no_prior_heterol_inf"].max() + 1))
for i, row in S_mapping.iterrows():
    D[i, int(row["no_prior_heterol_inf"])] = 1.0
D = pt.constant(D)

# Tutorials that helped build this model (step function)
# https://www.youtube.com/watch?v=G9VWXZdbtKQ
# https://pytensor.readthedocs.io/en/latest/library/scan.html 
# https://gist.github.com/ricardoV94/a49b2cc1cf0f32a5f6dc31d6856ccb63#file-pymc_timeseries_ma-ipynb
# https://becarioprecario.bitbucket.io/spde-gitbook/ch-intro.html

###############################
## Bayesian imputation model ##
###############################

def substep(beta_t, births_t, deaths_t, intro_mask_t, estimated_prop_t, f, f_per_I,
         S_t, I_t, P_t, 
         C, W, H_het, H_hom, L, K_het, K_hom,
         gamma, omega, birth_vec, dt):

    # --- Total population ---
    N_t = pt.sum(S_t, axis=1) + pt.sum(I_t, axis=1) + pt.sum(P_t, axis=1)   # (clusters,)
    
    # --- FOI ---
    lambda_t = beta_t[:, None] * (I_t @ C) / N_t[:, None]                   # (clusters, serotypes)
    lambda_t += 1e-7 * intro_mask_t[None, :] * estimated_prop_t             # seeding

    # --- Susceptibles ---
    effective_lambda = (lambda_t @ H_het.T + (lambda_t * f[None, :]) @ H_hom.T)     # (clusters, n_S)
    S_next = S_t + dt * (births_t[:, None] * birth_vec[None, :] - deaths_t[:, None] * S_t + (1/omega) * (P_t @ W) - effective_lambda * S_t)

    # --- Infectious  ---
    lambda_per_I = lambda_t @ L     # (clusters, n_I)
    S_to_I_het = S_t @ K_het.T                         
    S_to_I_hom = S_t @ K_hom.T                                
    delta_I_het = lambda_per_I * S_to_I_het                    
    delta_I_hom = lambda_per_I * (S_to_I_hom * f_per_I[None, :])
    I_next = I_t + dt * (-(1/gamma + deaths_t)[:, None] * I_t + delta_I_het + delta_I_hom)

    # --- Cross-protection ---
    P_next = P_t + dt * (-(1/omega + deaths_t)[:, None] * P_t + (1/gamma) * I_t)

    return S_next, I_next, P_next, dt * delta_I_hom, dt * delta_I_het, lambda_t


def step(beta_t, births_t, deaths_t, intro_mask_t, estimated_prop_t, f, f_per_I,
         S_t, I_t, P_t, 
         C, W, H_het, H_hom, L, K_het, K_hom,
         gamma, omega, birth_vec):

    # --- First substep ---
    S1, I1, P1, d_hom1, d_het1, _ = substep(
        beta_t, births_t, deaths_t, intro_mask_t, estimated_prop_t, f, f_per_I,
        S_t, I_t, P_t,
        C, W, H_het, H_hom, L, K_het, K_hom,
        gamma, omega, birth_vec,
        0.2
    )

    # --- Second substep ---
    S2, I2, P2, d_hom2, d_het2, _ = substep(
        beta_t, births_t, deaths_t, intro_mask_t, estimated_prop_t, f, f_per_I,
        S1, I1, P1,
        C, W, H_het, H_hom, L, K_het, K_hom,
        gamma, omega, birth_vec,
        0.2
    )

    # --- Third substep ---
    S3, I3, P3, d_hom3, d_het3, _ = substep(
        beta_t, births_t, deaths_t, intro_mask_t, estimated_prop_t, f, f_per_I, 
        S2, I2, P2,
        C, W, H_het, H_hom, L, K_het, K_hom,
        gamma, omega, birth_vec,
        0.2
    )

    # --- Fourth substep ---
    S4, I4, P4, d_hom4, d_het4, _ = substep(
        beta_t, births_t, deaths_t, intro_mask_t, estimated_prop_t, f, f_per_I,
        S3, I3, P3,
        C, W, H_het, H_hom, L, K_het, K_hom,
        gamma, omega, birth_vec,
        0.2
    )

    # --- Fifth substep ---
    S5, I5, P5, d_hom5, d_het5, lambda5 = substep(
        beta_t, births_t, deaths_t, intro_mask_t, estimated_prop_t, f, f_per_I,
        S4, I4, P4,
        C, W, H_het, H_hom, L, K_het, K_hom,
        gamma, omega, birth_vec,
        0.2
    )

    # --- accumulate Delta_I over the full month ---
    delta_hom_sum = d_hom1 + d_hom2 + d_hom3 + d_hom4 + d_hom5
    delta_het_sum = d_het1 + d_het2 + d_het3 + d_het4 + d_het5

    return S5, I5, P5, delta_hom_sum, delta_het_sum, lambda5


def build_initial_susceptibles(demo, f_P, pi_d, pi_mono2):
    """
    Construct initial susceptibles for 1999 (only DENV-1 and DENV-2)

    Parameters
    ----------
    demo : np.ndarray
        shape (n_clusters,)

    f_P: float
        fraction of total population in cross-protected state

    pi_d : TensorVariable
        shape (3,)
        dirichlet over degree of infection: [naive, mono, double]

    pi_mono12 : float
        splitting mono across DENV-1 DENV-2 [mono_1, mono_2]

    Returns
    -------
    S0 : TensorVariable
        shape (n_clusters, 16)
    """

    S_frac = pt.stack([pi_d[0], pi_d[1]*(1-pi_mono2), pi_d[1]*pi_mono2, 0, 0, pi_d[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return (1-f_P) * demo[:, None] * S_frac[None, :]


def build_initial_crossprotection(demo, f_P, pi_d, f_P2):
    """ See `build_initial_susceptibles` """

    # normalize pi_degree for degree of infection 1 and 2
    pi_sum = pi_d[1] + pi_d[2] + 1e-12  # prevent division by zero
    deg1_frac = pi_d[1] / pi_sum
    deg2_frac = pi_d[2] / pi_sum

    # set fractions
    P_frac = pt.stack([deg1_frac*(1-f_P2), deg1_frac*f_P2, 0, 0, deg2_frac * f_P2, 0, 0, deg2_frac * (1-f_P2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return f_P * demo[:, None] * P_frac[None, :]


def build_initial_infected(DENV_total, p0, kappa, pi_d):
    """
    Parameters
    ----------

    DENV_total: np.ndarray
        shape: (n_clusters,)
    
    p0: np.ndarray
        shape: (n_clusters, n_serotypes)

    kappa : TensorVariable
        shape (n_clusters, n_infection_degrees, n_serotypes, hom_het)
        fraction of infections observed
        
    pi_d : TensorVariable
        shape (3,)
        dirichlet over degree of infection: [naive, mono, double]

    Returns
    -------
    I0 : TensorVariable
        shape (n_clusters, 32)
    """

    # normalize pi_degree for degree of infection 1 and 2
    pi_sum = pi_d[0] + pi_d[1] + 1e-12  # prevent division by zero
    first_inf = pi_d[0] / pi_sum
    second_inf = pi_d[1] / pi_sum

    # divide the total dengue cases over the serotypes under the assumption they are heterologous infections
    DENV_total_est = ((DENV_total[:, None] * p0)[:, None, :] / kappa[:, :, :, 1]) * pt.stack([first_inf, second_inf, 0, 0])[None,:, None]   # shape: (n_clusters, n_serotypes)

    # zeros
    z = pt.zeros_like(DENV_total_est[:, 0, 0])
    return pt.stack([DENV_total_est[:, 0, 0], DENV_total_est[:, 0, 1], DENV_total_est[:, 0, 2], DENV_total_est[:, 0, 3], DENV_total_est[:, 1, 1], DENV_total_est[:, 1, 2], z, DENV_total_est[:, 1, 0], DENV_total_est[:, 1, 2],
                     z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z]).T


def ar1_step(eps_t, u_prev, rho, sigma_ar):
    u_t = rho * u_prev + sigma_ar * eps_t
    return u_t

with pm.Model() as model:

    # ----------------
    # Parameterisation
    # ----------------

    # initial states
    ## initial susceptible and cross-protected states (cluster x state_idx)
    f_P = 0.25 #pm.Beta("f_P", alpha=8, beta=24)                 # first division of cluster population happens based on amount in a cross-protected state
    f_P2 = 0.75 #pm.Beta("f_P2", alpha=3, beta=1)                # fraction cross-protected after DENV-2 infection
    pi_d = np.array([0.1, 0.2, 0.7]) #m.Dirichlet("pi_d", a=10*np.array([2, 4, 4]))   # divide the non-cross-protected across naive, mono, double
    pi_mono2 = 0.75 # pm.Beta("pi_mono2", alpha=3, beta=1)        # divide the mono between DENV-1 and DENV-2
    S0 = pm.Deterministic("S0", build_initial_susceptibles(demo, f_P, pi_d, pi_mono2))
    P0 = pm.Deterministic("P0", build_initial_crossprotection(demo, f_P, pi_d, f_P2))

    # Parameters 
    ## average duration of infection
    gamma = 1/2

    ## average duration cross-protection
    omega = 36 # pm.Lognormal("omega", mu=3, sigma=0.1)

    ## average FOI reduction for homologous infections (n_months x n_serotypes)
    ### time-dependent for DENV-1 / DENV-2
    #### DENV-1
    mu_f1 = pm.Beta("mu_f1", alpha=1, beta=10)
    rho_f1 = pm.Beta("rho_f1", alpha=3, beta=3)
    sigma_f1 = pm.HalfNormal("sigma_f1", sigma=1/3)
    eps_f1 = pm.Normal("eps_f1", 0, 1, shape=n_years+1)
    f1_logit_seq, _ = pytensor.scan(fn=ar1_step, sequences=eps_f1[1:], outputs_info=pt.zeros(()), non_sequences=[rho_f1, sigma_f1])
    f1_logit = pm.math.logit(mu_f1) + f1_logit_seq
    f1_t = pm.math.sigmoid(f1_logit[year_idx[::n_clusters]])
    #### DENV-2
    mu_f2 = pm.Beta("mu_f2", alpha=1, beta=10)
    rho_f2 = pm.Beta("rho_f2", alpha=3, beta=3)
    sigma_f2 = pm.HalfNormal("sigma_f2", sigma=1/3)
    eps_f2 = pm.Normal("eps_f2", 0, 1, shape=n_years+1)
    f2_logit_seq, _ = pytensor.scan(fn=ar1_step, sequences=eps_f2[1:], outputs_info=pt.zeros(()), non_sequences=[rho_f2, sigma_f2])
    f2_logit = pm.math.logit(mu_f2) + f2_logit_seq
    f2_t = pm.math.sigmoid(f2_logit[year_idx[::n_clusters]])
    ## time-independent for DENV-3
    f1 = 0.9
    f2 = 0.9
    f3 = 0.7 # pm.Beta("f3", alpha=1, beta=20)
    ## construct f & save it
    #f = pm.Deterministic("f", pt.stack([f1_t, f2_t, pt.repeat(f3, n_months), pt.repeat(0, n_months)], axis=1))
    f = pm.Deterministic("f", pt.stack([pt.repeat(f1, n_months), pt.repeat(f2, n_months), pt.repeat(f3, n_months), pt.repeat(0, n_months)], axis=1))
    ## construct f_per_I
    f_per_I = f[:, sero_idx]

    ## reported fraction (cluster x degree x serotype)
    kappa0_logit = pm.math.logit(1/10) #pm.Normal("kappa0_logit", mu=pm.math.logit(1/10), sigma=1)                 # intercept
    is_34 = pt.as_tensor([0,0,1,1])                                                             # indicator prim/sec versus tert/quart
    log_or_34 = pm.Normal("log_or_34", mu=-9, sigma=1/3)                                        # OR detecting prim/sec versus tert/quart
    log_or_serotype = pm.Normal("log_or_serotype", mu=0.0, sigma=1/3, shape=n_serotypes-1)    # OR detecting serotypes (vs. DENV-1)
    log_or_serotype_full = pt.concatenate([pt.zeros(1), log_or_serotype])
    log_or_cluster = pm.Normal("log_or_cluster", mu=0.0, sigma=1/3, shape=n_clusters-1)         # OR detecting in a cluster (vs. cluster 1)
    log_or_cluster_full = pt.concatenate([pt.zeros(1), log_or_cluster])
    is_hom = pt.as_tensor([1,0])                                                                # indicator for homologous infection
    log_or_hom = pm.Normal("log_or_hom", mu=-9, sigma=1/3)                                      #            
    logit_kappa = kappa0_logit + log_or_cluster_full[:, None, None, None] + log_or_34*is_34[None, :, None, None] \
        + log_or_serotype_full[None, None, :, None] + log_or_hom * is_hom[None, None, None, :]
    kappa = pm.Deterministic("kappa", pm.math.sigmoid(logit_kappa))
    or_34 = pm.Deterministic("or_34", pt.exp(log_or_34))
    or_cluster = pm.Deterministic("or_cluster", pt.exp(log_or_cluster_full))
    or_serotype = pm.Deterministic("or_serotype", pt.exp(log_or_serotype_full))
    or_homologous = pm.Deterministic("or_homologous", pt.exp(log_or_hom))

    ## Fixed components of the FOI: transmission coefficient beta (seasonal + AR(1); time x cluster)
    ### seasonal component
    mu_beta = np.log(2.5) * pt.ones(n_clusters) #pm.Normal("mu_beta", mu=np.log(2.5), sigma=1/5, shape=n_clusters)
    A_beta = 1 * pt.ones(n_clusters) # pm.HalfNormal("A_beta", sigma=1, shape=n_clusters)
    phi_beta = 1 * pt.ones(n_clusters) # pm.Normal("phi_beta", mu=pt.pi/3, sigma=1, shape=n_clusters) # peaks March
    season = A_beta[:, None] * pt.cos(2 * np.pi * pt.arange(n_months)[None, :] / 12 - phi_beta[:, None])
    ### AR(1) component (non-centered)
    rho_ar_beta = pm.Beta("rho_ar_beta", alpha=1, beta=3)                 # persistence
    sigma_ar_beta = pm.HalfNormal("sigma_ar_beta", sigma=1/10)             # freedom
    eps_raw = pm.Normal("eps_raw", mu=0, sigma=1, shape=(n_months, n_clusters))
    u_seq, _ = pytensor.scan(fn=ar1_step, sequences=eps_raw[1:], outputs_info=pt.zeros(n_clusters), non_sequences=[rho_ar_beta, sigma_ar_beta])
    u = pt.concatenate([ pt.zeros(n_clusters)[None, :], u_seq], axis=0)
    #### Combine
    beta_t = pm.Deterministic("beta_t", pt.exp(mu_beta[:, None] + season).T) #+ u.T
    
    ## Initial infected
    I0 = build_initial_infected(pt.as_tensor(DENV_total[0,:]), pt.as_tensor(p0.values), kappa, pi_d)

    # -----------------------------------------------------------------------------
    # Integrate transmission model (Euler's method; dt = 1 month; 5 steps per month
    # -----------------------------------------------------------------------------

    (S, I, P, Delta_I_hom, Delta_I_het, lambda_t), _ = pytensor.scan(
        fn=step,
        sequences=[beta_t,
                   pt.as_tensor_variable(births),
                   pt.as_tensor_variable(death_rate),
                   pt.as_tensor_variable(intro_mask),
                   pt.as_tensor_variable(estimated_proportions),
                   f, f_per_I
                   ],
        outputs_info=[S0, I0, P0, None, None, None],
        non_sequences=[
            C, W, H_het, H_hom, L, K_het, K_hom,
            gamma, omega, birth_vec
        ]
    )

    # ---------------------------
    # Derived simulation products
    # ---------------------------

    # compute FOI trajectory (time, cluster)
    lambda_t = pm.Deterministic("lambda_t", pt.sum(lambda_t, axis=-1)) 

    # reshape Delta_I into observations per (time, cluster, infection_degree, serotype, hom/het infection)
    # + soft bottom so Delta_I can never hit zero --> else p_reported = 0 --> a = 0 in DirichletMultinomial --> logp = -inf !!
    Delta_I = pm.Deterministic("Delta_I", 1 + pt.softplus(pt.stack([pt.einsum("tci,ids->tcds", Delta_I_hom, O), pt.einsum("tci,ids->tcds", Delta_I_het, O)], axis=-1) - 1))

    # compute "true" serotype proportions (time, cluster, serotype)
    I_sero = pt.dot(I, C)
    p = pm.Deterministic("p", (I_sero / pt.sum(I_sero, axis=2)[:, :, None]))

    # reshape S into susceptibility slots per (time, cluster, infection_degree, serotype and hom/het infection)
    S_expanded = pm.Deterministic("S_expanded", pt.stack([pt.einsum("tci,ids->tcds", S, R_het), pt.einsum("tci,ids,ts->tcds", S, R_hom, f)], axis=-1))

    # reshape S into susceptible individuals per (time, degree)
    S_degree = pm.Deterministic("S_degree", pt.einsum("tci,id->tcd", S, D))

    # -----------
    # Observation
    # -----------

    # Observed cases
    reported = pt.sum(Delta_I * kappa, axis=(2,3,4))
    alpha_inv = pm.HalfNormal("alpha_inv", sigma=1/10)
    D_obs = pm.NegativeBinomial("D_obs", mu=reported, alpha=1/alpha_inv, observed=DENV_total)

    # Observed serotyped cases
    ## Hierarchical overdispersion (per cluster)
    d_cluster_hierarch = pm.HalfNormal("d_cluster_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
    d_cluster = pm.HalfNormal("d_cluster", sigma=d_cluster_hierarch, shape=n_clusters)
    phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_cluster, 1e-12))[None, :], n_months, axis=0))
    reported_serotype = pt.sum(Delta_I * kappa, axis=(2,4))
    p_detect = pm.Deterministic("p_detect", reported_serotype / pt.sum(reported_serotype, axis=2, keepdims=True))
    alpha = phi[:, :, None] * p_detect
    VIF = pm.Deterministic("VIF", (N_typed + phi) / (1 + phi)) # variance inflation of dirichlet multinomial compared to multinomial

    # ad-hoc visualisation
    fig,ax=plt.subplots(nrows=6)
    # serotype proportions
    ax[0].stackplot(range(len(p.eval()[:,0,0])),
                        p.eval()[:,0,0], 
                        p.eval()[:,0,1],
                        p.eval()[:,0,2],
                        p.eval()[:,0,3],
                        colors=['black', 'red', 'green', 'blue'])
    # detected serotype proportions
    ax[1].stackplot(range(len(p_detect.eval()[:,0,0])),
                        p_detect.eval()[:,0,0], 
                        p_detect.eval()[:,0,1],
                        p_detect.eval()[:,0,2],
                        p_detect.eval()[:,0,3],
                        colors=['black', 'red', 'green', 'blue'])
    # observed cases
    ax[2].scatter(range(len(DENV_total[:,0])), DENV_total[:,0], color='black', alpha=0.6, s=5)
    ax[2].plot(D_obs.eval()[:,0], color='red')
    # force of infection
    ax[3].plot(lambda_t.eval()[:,0], color='black')
    ax[3].axhline(np.mean(lambda_t.eval()[:,0]), color='red')
    # susceptibility slots (total)
    ax[4].plot(pt.sum(S_expanded, axis=(2,4)).eval()[:,0,0], color='black')
    ax[4].plot(pt.sum(S_expanded, axis=(2,4)).eval()[:,0,1], color='red')
    ax[4].plot(pt.sum(S_expanded, axis=(2,4)).eval()[:,0,2], color='green')
    ax[4].plot(pt.sum(S_expanded, axis=(2,4)).eval()[:,0,3], color='blue')
    # susceptibility slots  (homologous)
    ax[4].plot(pt.sum(S_expanded, axis=2).eval()[:,0,0,0], color='black', linestyle='dashed')
    ax[4].plot(pt.sum(S_expanded, axis=2).eval()[:,0,1,0], color='red', linestyle='dashed')
    ax[4].plot(pt.sum(S_expanded, axis=2).eval()[:,0,2,0], color='green', linestyle='dashed')
    ax[4].plot(pt.sum(S_expanded, axis=2).eval()[:,0,3,0], color='blue', linestyle='dashed')
    # number of susceptible individuals per degree of infection
    ax[5].stackplot(range(len(S_degree.eval()[:,0,0])),
                            S_degree.eval()[:,0,0],
                            S_degree.eval()[:,0,1],
                            S_degree.eval()[:,0,2],
                            S_degree.eval()[:,0,3],
                            S_degree.eval()[:,0,4],
                            colors=['black', 'red', 'orange', 'yellow', 'green'])
    plt.show()
    plt.close()

    # --- Observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial)

#######################
## Running the model ##
#######################

# NUTS
draws=250
with model:
    trace = pm.sample(draws, tune=250, target_accept=0.99,
                     chains=chains, cores=chains, init='adapt_diag', progressbar=True,
                     initvals=chains*[{'f_P': 0.2, 'f_P2': 0.75, 'pi_d': pt.as_tensor([0.2, 0.4, 0.4]), 'pi_mono2': 0.75,
                                       'omega': 12, 'mu_f1': 0.8, 'mu_f2': 0.8, 'f3': 0.5, 'kappa0_logit': pm.math.logit(0.1),
                                       'mu_beta': np.log(2) * pt.ones(n_clusters), 'A_beta': 1 * pt.ones(n_clusters), 'phi_beta': pt.ones(n_clusters),
                                       'alpha_inv': 0.3}],
                     idata_kwargs={'log_likelihood':True})

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

# Assume `trace` is the result of pm.sample()
arviz.to_netcdf(trace, f"{output_folder}/trace.nc")
arviz.to_netcdf(posterior_predictive, f"{output_folder}/posterior_predictive.nc")

# Traceplot
variables2plot = [
                 'f_P', 'f_P2', 'pi_d', 'pi_mono2', 'omega', 'mu_f1', 'sigma_f1', 'rho_f1', 'mu_f2', 'sigma_f2', 'rho_f2', 'f3', 'kappa', 'kappa0_logit', 'or_34', 'or_cluster', 'or_serotype', 'or_homologous', 'mu_beta', 'A_beta', 'phi_beta', 'rho_ar_beta', 'sigma_ar_beta', 'alpha_inv', 'd_cluster_hierarch', 'd_cluster',
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



