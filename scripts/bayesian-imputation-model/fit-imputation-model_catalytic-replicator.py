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
start_year = 1999
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

# only do first X clusters
df = df[df['cluster'].isin([4,5])]

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
demo = demo[((demo['year'] == start_year) & (demo['cluster'].isin([4,5])))]['population'].values 

# --- Indices ---
cluster_idx = df["cluster"].to_numpy().astype(int)
month_idx = df["month_idx"].to_numpy().astype(int)
year_idx = df["year_idx"].to_numpy().astype(int)

# --- Lengths ---
n_clusters = int(len(df['cluster'].unique()))
n_months = int(len(df["month_idx"].unique()))
n_years = int(df["year_idx"].max() + 1)
n_serotypes = len(sero_cols)

# Estimate initial serotype distribution p0 from first year of data in every cluster
# Use the mean of the posterior of Dirichlet-Multinomial model with symmetric prior alpha = 1/2 (Jeffrey's prior = uninformative prior)
d = df[df['year'] == min(df['year'])].groupby(by='cluster')[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].sum()
cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4"]
alpha = 1/2
p0 = (d[cols] + alpha).div(d[cols].sum(axis=1) + len(cols) * alpha, axis=0)

# Make a serotype introduction mask; shape: (n_months, n_serotypes)
unique_dates = np.sort(df["date"].unique())
years = pd.DatetimeIndex(unique_dates).year
intro_mask = np.ones((len(unique_dates), 4))
intro_mask[:, 3] = (years >= 2007).astype(int)


# Tutorials that helped build this model (step function)
# https://www.youtube.com/watch?v=G9VWXZdbtKQ
# https://pytensor.readthedocs.io/en/latest/library/scan.html 
# https://gist.github.com/ricardoV94/a49b2cc1cf0f32a5f6dc31d6856ccb63#file-pymc_timeseries_ma-ipynb
# https://becarioprecario.bitbucket.io/spde-gitbook/ch-intro.html

###############################
## Bayesian imputation model ##
###############################

def update_susceptibles(S, P, lambd, p, births, deaths, omega):
    """
    
    input
    -----

    S: TensorVariable
        shape: (n_clusters, state_idx)

    P: TensorVariable
        shape: (n_clusters, state_idx)
    
    lambd(a): TensorVariable
        shape: (n_clusters,)

    p: TensorVariable
        shape: (n_clusters, n_serotypes)
    
    births: TensorVariable
        shape: (n_clusters,)

    deaths: TensorVariable
        shape: (n_clusters,)
    
    omega: float
    """

    # ---- immune naive ----
    col0 = (1 - deaths) * S[:, 0] + births - pt.sum(p * lambd[:, None] * S[:, 0, None], axis=1)

    # ---- 1 prior infection ----
    col1 = (1 - deaths) * S[:, 1] + (1/omega) * P[:, 0] - pt.sum(p[:, (1,2,3)] * lambd[:, None] * S[:, 1, None], axis=1)
    col2 = (1 - deaths) * S[:, 2] + (1/omega) * P[:, 1] - pt.sum(p[:, (0,2,3)] * lambd[:, None] * S[:, 2, None], axis=1)
    col3 = (1 - deaths) * S[:, 3] + (1/omega) * P[:, 2] - pt.sum(p[:, (0,1,3)] * lambd[:, None] * S[:, 3, None], axis=1)
    col4 = (1 - deaths) * S[:, 4] + (1/omega) * P[:, 3] - pt.sum(p[:, (0,1,2)] * lambd[:, None] * S[:, 4, None], axis=1)

    # ---- 2 prior infections ----
    col5 = (1 - deaths) * S[:, 5] + (1/omega) * (P[:, 4] + P[:, 7]) - pt.sum(p[:, (2,3)] * lambd[:, None] * S[:, 5, None], axis=1)
    col6 = (1 - deaths) * S[:, 6] + (1/omega) * (P[:, 5] + P[:, 10]) - pt.sum(p[:, (1,3)] * lambd[:, None] * S[:, 6, None], axis=1)
    col7 = (1 - deaths) * S[:, 7] + (1/omega) * (P[:, 6] + P[:, 13]) - pt.sum(p[:, (1,2)] * lambd[:, None] * S[:, 7, None], axis=1)
    col8 = (1 - deaths) * S[:, 8] + (1/omega) * (P[:, 8] + P[:, 11]) - pt.sum(p[:, (0,3)] * lambd[:, None] * S[:, 8, None], axis=1)
    col9 = (1 - deaths) * S[:, 9] + (1/omega) * (P[:, 9] + P[:, 14]) - pt.sum(p[:, (0,2)] * lambd[:, None] * S[:, 9, None], axis=1)
    col10 = (1 - deaths) * S[:, 10] + (1/omega) * (P[:, 12] + P[:, 15]) - pt.sum(p[:, (0,1)] * lambd[:, None] * S[:, 10, None], axis=1)

    # ---- 3 prior infections ----
    col11 = (1 - deaths) * S[:, 11] + (1/omega) * (P[:, 27] + P[:, 25] + P[:, 23]) - p[:, 0] * lambd * S[:, 11]
    col12 = (1 - deaths) * S[:, 12] + (1/omega) * (P[:, 26] + P[:, 21] + P[:, 19]) - p[:, 1] * lambd * S[:, 12]
    col13 = (1 - deaths) * S[:, 13] + (1/omega) * (P[:, 24] + P[:, 20] + P[:, 17]) - p[:, 2] * lambd * S[:, 13]
    col14 = (1 - deaths) * S[:, 14] + (1/omega) * (P[:, 22] + P[:, 18] + P[:, 16]) - p[:, 3] * lambd * S[:, 14]

    # ---- recovered ----
    col15 = (1 - deaths) * S[:, 15] + p[:, 0] * lambd * S[:, 11] + p[:, 1] * lambd * S[:, 12] + p[:, 2] * lambd * S[:, 13] + p[:, 3] * lambd * S[:, 14]

    return pt.stack([col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15], axis=1)



def update_cross_protection(S, P, lambd, p, deaths, omega):
    """See `update_susceptibles` """

    # ---- post first infection ----
    col0 = (1 - deaths - 1/omega) * P[:, 0] + p[:, 0] * lambd * S[:, 0]
    col1 = (1 - deaths - 1/omega) * P[:, 1] + p[:, 1] * lambd * S[:, 0]
    col2 = (1 - deaths - 1/omega) * P[:, 2] + p[:, 2] * lambd * S[:, 0]
    col3 = (1 - deaths - 1/omega) * P[:, 3] + p[:, 3] * lambd * S[:, 0]

    # ---- post second infection ----
    ## first infection with 1
    col4 = (1 - deaths - 1/omega) * P[:, 4] + p[:, 1] * lambd * S[:, 1]
    col5 = (1 - deaths - 1/omega) * P[:, 5] + p[:, 2] * lambd * S[:, 1]
    col6 = (1 - deaths - 1/omega) * P[:, 6] + p[:, 3] * lambd * S[:, 1]
    ## first infection with 2
    col7 = (1 - deaths - 1/omega) * P[:, 7] + p[:, 0] * lambd * S[:, 2]
    col8 = (1 - deaths - 1/omega) * P[:, 8] + p[:, 2] * lambd * S[:, 2]
    col9 = (1 - deaths - 1/omega) * P[:, 9] + p[:, 3] * lambd * S[:, 2]
    ## first infection with 3
    col10 = (1 - deaths - 1/omega) * P[:, 10] + p[:, 0] * lambd * S[:, 3]
    col11 = (1 - deaths - 1/omega) * P[:, 11] + p[:, 1] * lambd * S[:, 3]
    col12 = (1 - deaths - 1/omega) * P[:, 12] + p[:, 3] * lambd * S[:, 3]
    ## first infection with 4
    col13 = (1 - deaths - 1/omega) * P[:, 13] + p[:, 0] * lambd * S[:, 4]
    col14 = (1 - deaths - 1/omega) * P[:, 14] + p[:, 1] * lambd * S[:, 4]
    col15 = (1 - deaths - 1/omega) * P[:, 15] + p[:, 2] * lambd * S[:, 4]    

    # ---- post third infection ----
    ## previously infected with 1 and 2
    col16 = (1 - deaths - 1/omega) * P[:, 16] + p[:, 2] * lambd * S[:, 5]    
    col17 = (1 - deaths - 1/omega) * P[:, 17] + p[:, 3] * lambd * S[:, 5]  
    ## previously infected with 1 and 3
    col18 = (1 - deaths - 1/omega) * P[:, 18] + p[:, 1] * lambd * S[:, 6]  
    col19 = (1 - deaths - 1/omega) * P[:, 19] + p[:, 3] * lambd * S[:, 6]  
    ## previously infected with 1 and 4
    col20 = (1 - deaths - 1/omega) * P[:, 20] + p[:, 1] * lambd * S[:, 7]
    col21 = (1 - deaths - 1/omega) * P[:, 21] + p[:, 2] * lambd * S[:, 7]
    ## previously infected with 2 and 3
    col22 = (1 - deaths - 1/omega) * P[:, 22] + p[:, 0] * lambd * S[:, 8]
    col23 = (1 - deaths - 1/omega) * P[:, 23] + p[:, 3] * lambd * S[:, 8]
    ## previously infected with 2 and 4
    col24 = (1 - deaths - 1/omega) * P[:, 24] + p[:, 0] * lambd * S[:, 9]
    col25 = (1 - deaths - 1/omega) * P[:, 25] + p[:, 2] * lambd * S[:, 9]
    ## previously infected with 3 and 4
    col26 = (1 - deaths - 1/omega) * P[:, 26] + p[:, 0] * lambd * S[:, 10]
    col27 = (1 - deaths - 1/omega) * P[:, 27] + p[:, 1] * lambd * S[:, 10]

    return pt.stack([col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15,
                     col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27], axis=1)



def get_susceptibles_serotype(S):
    """
    S: TensorVariable of shape (n_clusters, state_idx)

    Returns
    -------
    TensorVariable of shape (n_clusters, n_serotypes)
    """

    S1 = pt.sum(S[:, (0, 2, 3, 4, 8, 9, 10, 11)], axis=1)
    S2 = pt.sum(S[:, (0, 1, 3, 4, 6, 7, 10, 12)], axis=1)
    S3 = pt.sum(S[:, (0, 1, 2, 4, 5, 7, 9, 13)], axis=1)
    S4 = pt.sum(S[:, (0, 1, 2, 3, 5, 6, 8, 14)], axis=1)

    return pt.stack([S1, S2, S3, S4], axis=1)



def get_susceptibles_serotype_time(S):
    """
    S: TensorVariable of shape (n_times, n_clusters, state_idx)

    Returns
    -------
    TensorVariable of shape (n_times, n_clusters, n_serotypes)
    """

    S1 = pt.sum(S[:, :, [0, 2, 3, 4, 8, 9, 10, 11]], axis=2)
    S2 = pt.sum(S[:, :, [0, 1, 3, 4, 6, 7, 10, 12]], axis=2)
    S3 = pt.sum(S[:, :, [0, 1, 2, 4, 5, 7, 9, 13]], axis=2)
    S4 = pt.sum(S[:, :, [0, 1, 2, 3, 5, 6, 8, 14]], axis=2)

    return pt.stack([S1, S2, S3, S4], axis=-1)



def get_susceptibles_by_degree_serotype(S):
    """
    S : TensorVariable of shape = (n_times, n_clusters, state_idx)

    Returns
    -------
    TensorVariable of shape = (n_times, n_clusters, n_degree_infection, n_serotypes)
    """

    # ---- degree 0 (naive) ----
    # susceptible to all 4 serotypes
    deg0 = pt.stack([S[:, :, 0]] * 4, axis=-1)

    # ---- degree 1 ----
    deg1 = pt.stack([
        S[:, :, 2] + S[:, :, 3] + S[:, :, 4],  # D1
        S[:, :, 1] + S[:, :, 3] + S[:, :, 4],  # D2
        S[:, :, 1] + S[:, :, 2] + S[:, :, 4],  # D3
        S[:, :, 1] + S[:, :, 2] + S[:, :, 3],  # D4
    ], axis=-1)

    # ---- degree 2 ----
    deg2 = pt.stack([
        S[:, :, 8] + S[:, :, 9] + S[:, :, 10],   
        S[:, :, 6] + S[:, :, 7] + S[:, :, 10],   
        S[:, :, 5] + S[:, :, 7] + S[:, :, 9],    
        S[:, :, 5] + S[:, :, 6] + S[:, :, 8],   
    ], axis=-1)

    # ---- degree 3 ----
    deg3 = pt.stack([
        S[:, :, 11],
        S[:, :, 12],
        S[:, :, 13],
        S[:, :, 14],
    ], axis=-1)

    return pt.stack([deg0, deg1, deg2, deg3], axis=2)



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
    pi_sum = pi_d[0] + pi_d[1] + 1e-12  # prevent division by zero
    deg1_frac = pi_d[0] / pi_sum
    deg2_frac = pi_d[1] / pi_sum

    # set fractions
    P_frac = pt.stack([deg1_frac*(1-f_P2), deg1_frac*f_P2, 0, 0, deg2_frac * f_P2, 0, 0, deg2_frac * (1-f_P2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return f_P * demo[:, None] * P_frac[None, :]

def ar1_step(eps_t, u_prev, rho, sigma_ar):
    u_t = rho * u_prev + sigma_ar * eps_t
    return u_t

# ideas:
# bounded influence of susceptibility differences on fitness through a tanh structure
# shift domain kappa to [0.002, 1] (hard lower bound to prevent bad geometries for NUTS sampler)

with pm.Model() as model:

    # ----------------
    # Parameterisation
    # ----------------

    # initial states
    ## initial susceptible and cross-protected states (cluster x state_idx)
    f_P = pm.Beta("f_P", alpha=10, beta=20)             # first division of cluster population happens based on amount in a cross-protected state
    f_P2 = pm.Beta("f_P2", alpha=30, beta=1)            # fraction cross-protected after DENV-2 infection
    pi_d = pm.Dirichlet("pi_d", a=[1, 3, 6])            # divide the non-cross-protected across naive, mono, double
    pi_mono2 = pm.Beta("pi_mono2", alpha=30, beta=1)    # divide the mono between DENV-1 and DENV-2
    S0 = pm.Deterministic("S0", build_initial_susceptibles(demo, f_P, pi_d, pi_mono2))
    P0 = pm.Deterministic("P0", build_initial_crossprotection(demo, f_P, pi_d, f_P2))

    ## initial gap between DENV-4 and the rest in z-transformed space 
    zgap_DENV4 = pm.Normal("zgap_DENV4", mu=-15, sigma=3, shape=n_clusters)
    z0 = pt.set_subtensor(pt.log(p0)[:, 3], pt.min(pt.log(p0)[:, :3], axis=1) + zgap_DENV4)


    # Parameters 
    ## fitness model
    beta = pm.Lognormal("beta", mu=pt.log(0.1), sigma=0.2)

    ## average duration cross-protection
    omega = pm.Lognormal("omega", mu=2.45, sigma=1/3)

    ## reported fraction (cluster x degree x serotype)
    kappa0_logit = pm.Normal("kappa0_logit", mu=pm.math.logit(1/10), sigma=1.0)             # intercept
    is_12 = pt.as_tensor([1, 1, 0, 0])                                                      # indicator of prim./sec. infection
    log_or_12 = pm.Normal("log_or_12", mu=0.0, sigma=1/3)                                   # OR detecting prim./sec. heterologous infection (vs. tert, quart)
    log_or_serotype = pm.Normal("log_or_serotype", mu=[0, -2, 0], sigma=1.0, shape=n_serotypes-1)  # OR detecting serotypes (vs. DENV-1)
    log_or_serotype_full = pt.concatenate([pt.zeros(1), log_or_serotype])
    log_or_cluster = pm.Normal("log_or_cluster", mu=0.0, sigma=1.0, shape=n_clusters-1)     # OR detecting in a cluster (vs. cluster 1)
    log_or_cluster_full = pt.concatenate([pt.zeros(1), log_or_cluster])                     
    logit_kappa = (kappa0_logit + log_or_cluster_full[:, None, None] + log_or_12 * is_12[None, :, None] + log_or_serotype_full[None, None, :])
    kappa = pm.Deterministic("kappa", pm.math.sigmoid(logit_kappa))
    or_cluster = pm.Deterministic("or_cluster", pt.exp(log_or_cluster_full))
    or_12 = pm.Deterministic("or_12", pt.exp(log_or_12))
    or_serotype = pm.Deterministic("or_serotype", pt.exp(log_or_serotype_full))

    ## Total FOI (seasonal + AR(1); time x cluster)
    ### seasonal component
    mu_lambda = pm.Normal("mu_lambda", mu=-3, sigma=1/3, shape=n_clusters)
    A_lambda = pm.HalfNormal("A_lambda", sigma=1, shape=n_clusters)
    phi_lambda = pm.Normal("phi_lambda", mu=pt.pi/3, sigma=1, shape=n_clusters) # peaks March
    season = A_lambda[:, None] * pt.cos(2 * np.pi * pt.arange(n_months)[None, :] / 12 - phi_lambda[:, None])
    ### AR(1) component (non-centered)
    rho_ar_lambda = pm.Beta("rho_ar_lambda", alpha=1, beta=2)                 # persistence
    sigma_ar_lambda = pm.HalfNormal("sigma_ar_lambda", sigma=1/3)             # freedom
    eps_raw = pm.Normal("eps_raw", mu=0, sigma=1, shape=(n_months, n_clusters))
    u_seq, _ = pytensor.scan(fn=ar1_step, sequences=eps_raw[1:], outputs_info=pt.zeros(n_clusters), non_sequences=[rho_ar_lambda, sigma_ar_lambda])
    u = pt.concatenate([ pt.zeros(n_clusters)[None, :], u_seq], axis=0)
    #### Combine
    log_lambda = mu_lambda[:, None] + season + u.T
    lambda_t = pm.Deterministic("lambda_t", pt.exp(log_lambda).T)

    ## Residual (RW(1); time x cluster)
    #sigma_residual = pm.HalfNormal("sigma_residual", sigma=0.0001)
    #eta =  pm.Normal("eta", mu=0.0, sigma=sigma_residual, shape=(n_months-1, n_clusters, n_serotypes))  
    eps0 = pt.zeros((n_clusters, n_serotypes))
    beta_f0 = pt.zeros((n_clusters, n_serotypes))

    # ---------------------------------------------------------------------
    # (Logit) Replicator equation integrated using Euler's method with dt=1
    # ---------------------------------------------------------------------

    def step(births_t, deaths_t, lambda_t, intro_mask_t, S_prev, P_prev, z_prev, beta_f_prev, eps_prev, beta, omega):
        
        p_prev = pm.math.softmax(z_prev, axis=1)                # reconstruct p

        # 1. Catalytic model
        S_t = update_susceptibles(S_prev, P_prev, lambda_t, p_prev, births_t, deaths_t, omega)
        P_t = update_cross_protection(S_prev, P_prev, lambda_t, p_prev, deaths_t, omega)
        S_t = 1 + pt.softplus(S_t - 1)
        P_t = 1 + pt.softplus(P_t - 1)

        # 2. Get susceptibles per serotype and cluster
        S_serotype = get_susceptibles_serotype(S_t)

        # 3. Compute fitness
        f_t = intro_mask_t * pt.log(S_serotype / pt.mean(S_serotype, axis=1, keepdims=True))

        # 4. RW(1) residual
        eps_t = eps_prev #+ intro_mask_t * eta_t

        # 4. Replicator update + residual
        z_t = z_prev + beta * f_t + eps_t

        return S_t, P_t, z_t, beta*f_t, eps_t

    # run step sequence
    (S_seq, P_seq, z_seq, beta_f_seq, eps_seq), _ = pytensor.scan(
        fn=step,
        sequences=[
            pt.as_tensor_variable(births),
            pt.as_tensor_variable(death_rate),
            lambda_t,
            pt.as_tensor_variable(intro_mask)
        ],
        outputs_info=[S0, P0, z0, beta_f0, eps0], # start_year = 2001 means first datapoint = 2001-01-31 --> The state '0' is 2000-12-31
        non_sequences=[beta, omega],
    )

    # attach initial states
    S = pm.Deterministic("S", pt.concatenate([S0[None, :, :], S_seq], axis=0))
    P = pm.Deterministic("P", pt.concatenate([P0[None, :, :], P_seq], axis=0))
    z = pm.Deterministic("z", pt.concatenate([pt.log(p0)[None, :, :], z_seq], axis=0))
    beta_f = pm.Deterministic("beta_f", pt.concatenate([beta_f0[None, :, :], beta_f_seq], axis=0))
    p = pm.Deterministic("p", pm.math.softmax(z, axis=2))
    eps = pm.Deterministic("eps", pt.concatenate([eps0[None, :, :], eps_seq], axis=0))

    # -----------
    # Observation
    # -----------

    # Observed cases
    alpha_inv = pm.HalfNormal("alpha_inv", sigma=1/10)
    S_degree_serotype = get_susceptibles_by_degree_serotype(S)
    infections = lambda_t[:, :, None, None] * p[:, :, None, :] * S_degree_serotype
    reported = infections * kappa[None, :, :, :]
    D_obs = pm.NegativeBinomial("D_obs", mu=pt.sum(reported, axis=(2, 3)), alpha=1/alpha_inv, observed=DENV_total)


    S_star = get_susceptibles_serotype_time(S)
    fig,ax=plt.subplots(nrows=4)
    # serotype proportions
    ax[0].plot(p.eval()[:,0,0], color='black')
    ax[0].plot(p.eval()[:,0,1], color='red')
    ax[0].plot(p.eval()[:,0,2], color='green')
    ax[0].plot(p.eval()[:,0,3], color='blue')
    # susceptibles
    ax[1].plot(S_star.eval()[:,0,0], color='black')
    ax[1].plot(S_star.eval()[:,0,1], color='red')
    ax[1].plot(S_star.eval()[:,0,2], color='green')
    ax[1].plot(S_star.eval()[:,0,3], color='blue')
    # observed cases
    ax[2].scatter(range(len(DENV_total[:,0])), DENV_total[:,0], color='black', alpha=0.6, s=5)
    ax[2].plot(D_obs.eval()[:,0], color='red')
    # force of infection
    ax[3].plot(lambda_t.eval()[:,0], color='red')
    plt.show()
    plt.close()


    # Observed serotyped cases
    ## Hierarchical overdispersion (per cluster)
    d_cluster_hierarch = pm.HalfNormal("d_cluster_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
    d_cluster = pm.HalfNormal("d_cluster", sigma=d_cluster_hierarch, shape=n_clusters)
    phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_cluster, 1e-12))[None, :], n_months, axis=0))
    reported_serotype = pt.sum(reported, axis=2)
    p_detect = reported_serotype / pt.sum(reported_serotype, axis=2, keepdims=True)
    alpha = phi[:, :, None] * p_detect
    VIF = pm.Deterministic("VIF", (N_typed + phi) / (1 + phi)) # variance inflation of dirichlet multinomial compared to multinomial

    # --- Observed subtyped incidences ---
    Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial)

#######################
## Running the model ##
#######################

# NUTS
draws=10
with model:
    trace = pm.sample(draws, tune=10, target_accept=0.99, chains=chains, cores=chains, init='adapt_diag', progressbar=True, idata_kwargs={'log_likelihood':True})

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
                 'zgap_DENV4', 'beta', 'kappa', 'kappa0_logit', 'or_cluster', 'or_12', 'or_serotype', 'pi_d', 'f_P', 'f_P2', 'pi_mono2', 'omega', 'mu_lambda', 'A_lambda', 'phi_lambda', 'rho_ar_lambda', 'sigma_ar_lambda', 'alpha_inv', 'd_cluster_hierarch', 'd_cluster',
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



