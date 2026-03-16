import os
import arviz
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt

# analysis startdate
start_year = 2001
end_year = 2024
assert start_year >= 1996, "earliest start_year is 1996."

# helper function for argument parsing
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# arguments are used to find the result
# How to run: python visualise-fit.py -date 2025-08-27 -ID test -p 2 -distance_matrix False
parser = argparse.ArgumentParser()
parser.add_argument("-ID", type=str, help="Identifier of the pipeline run.")
parser.add_argument("-spatial_aggregation", type=str, help="Spatial aggregation clustering was performed on.")
#parser.add_argument("-p", type=int, help="Order of AR(p) process.", default=1)
#parser.add_argument("-q", type=int, help="Order of MA(q) process.", default=1)
args = parser.parse_args()

# assign to desired variables
spatial_aggregation = args.spatial_aggregation
ID = args.ID
#p = args.p
#q = args.q

# pipeline output folder
abs_dir = os.path.dirname(__file__) # make sure all referenced paths are relative to the lcoation of this file and not the terminal's pwd
output_folder = os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/bayesian-imputation-model_output/new_model/')
# check if output dir exists, if not, raise an error
if not os.path.exists(output_folder):
    raise ValueError('result not found.')

##############
## Settings ##
##############

# set confidence level
confidence = 95

###################################################
## Get the traces and posterior predictive check ##
###################################################

# Load the trace from a NetCDF file
trace = arviz.from_netcdf(f"{output_folder}/trace.nc")

# Load the posterior samples
posterior_predictive = arviz.from_netcdf(f"{output_folder}/posterior_predictive.nc")

###########################################################################
## Preparing the incidence data (excludes within-sample validation data) ##
###########################################################################

# Load left out spatial units
validation_labels = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/clusters/validation_labels.csv')).squeeze()

# Load clusters
clusters = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/clusters/clusters_{spatial_aggregation}.csv'))
region = clusters.columns.to_list()[0]

# Load mapping
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))
mapping = mapping.merge(clusters[[region, 'cluster']], on=region, how='left')

# Compute demography in start_year per cluster
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

demo = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/IBGE_population-projections/pop-births-deaths_mun_1996-2024.csv'))
demo = demo.rename(columns={'geocode': 'CD_MUN'})
demo = demo.merge(mapping[['CD_MUN', 'cluster']], on='CD_MUN', how='left')
demo = demo.groupby(['cluster', 'year'], as_index=False)['population'].sum()
demo = demo[demo['year'] == start_year]

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
## Save validation data
df_validation = df.loc[df['CD_MUN'].isin(validation_labels.values)]
## Add the cluster label
mapping = mapping[['CD_MUN', f'{region}']]
mapping = clusters.merge(mapping, on=f'{region}', how="left")
df_validation = df_validation.merge(mapping[['CD_MUN', 'cluster']], on='CD_MUN', how='left')
## Remove from visualisation dataset
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


################################
## Output the imputed dataset ##
################################

# add latent serotype ratios to output dataframe
output = df[['date', 'cluster', 'DENV_1', 'DENV_2', 'DENV_3', 'DENV_4', 'DENV_total']]
output[['p_1', 'p_2', 'p_3', 'p_4']] = trace['posterior']['p'].mean(dim=['chain','draw']).values.reshape((n_months*n_clusters, n_serotypes))
output  = output[["date", "cluster", "p_1", "p_2", "p_3", "p_4"]]
output_mun = output.merge(mapping[["cluster", "CD_MUN"]], on="cluster", how="left")
output_mun = output_mun[["date", "CD_MUN", "p_1", "p_2", "p_3", "p_4"]]
output_mun = output_mun.sort_values(by=["date", "CD_MUN"]).reset_index(drop=True)
# refetch incidence data & merge it
inc = pd.read_csv(os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv'), parse_dates=['date'])
output_mun = output_mun.merge(
    inc[['CD_MUN', 'date', 'DENV_total']],
    on=['CD_MUN', 'date'],
    how='left'  # keeps all rows from df2
)
# fetch geodata to get population count
geography = gpd.read_parquet(os.path.join(abs_dir, "../../data/interim/geographic-dataset.parquet"))
output_mun = output_mun.merge(
    geography[['CD_MUN', 'POP']],
    on='CD_MUN',
    how='left'
)

# save result
output_mun.to_parquet(f'{output_folder}/DENV-serotypes-imputed_1996-2025_monthly.parquet', compression='brotli')

##############################################
## Compute within-sample validation metrics ##
##############################################

# Extract p_i per municipality
df_p = output_mun.loc[output_mun['CD_MUN'].isin(validation_labels.values)][['date', 'CD_MUN', 'p_1', 'p_2', 'p_3', 'p_4']]

# Extract phi per cluster
phi_mean = df[['cluster','date']]
phi_mean['phi'] = trace['posterior']['phi'].mean(dim=['chain','draw']).values.flatten()
phi_mean = phi_mean.set_index(['cluster','date'])
phi_mean = phi_mean.groupby(by='cluster')['phi'].mean()

# Merge p_i and phi --> compute alpha_i = p_i * phi
df_p = df_p.merge(mapping[['CD_MUN', 'cluster']], on='CD_MUN', how='left')
df_p = df_p.merge(phi_mean, on='cluster', how='left')
df_p[['a_1', 'a_2', 'a_3', 'a_4']] = df_p[['p_1', 'p_2', 'p_3', 'p_4']].values * df_p[['phi']].values

# Only evaluate log-likelihood when data are valid
mask = ~df_validation[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].isna().all(axis=1)
df_validation = df_validation.dropna(subset=['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4'], how='all')
df_validation = df_validation.fillna(0)
df_p = df_p.loc[mask]

# Compute log likelihood
from scipy.stats import dirichlet_multinomial
x = df_validation[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].values
n = np.sum(x, axis=1)
alpha = df_p[['a_1', 'a_2', 'a_3', 'a_4']].values
logp = dirichlet_multinomial.logpmf(x=x, alpha=alpha, n=n)

# Save result
df_validation['ll_dirichletmultinomial'] = logp
df_validation[['p_1', 'p_2', 'p_3', 'p_4']] = df_p[['p_1', 'p_2', 'p_3', 'p_4']].values
df_validation['phi'] = df_p[['phi']].values
df_validation[['date', 'CD_MUN', 'cluster', 'phi', 'DENV_1', 'DENV_2', 'DENV_3', 'DENV_4', 'p_1', 'p_1', 'p_3', 'p_4', 'll_dirichletmultinomial']].to_csv(f'{output_folder}/validation_loglikelihood.csv', index=False)


################################################
## Prepare the data used in the visualisation ##
################################################

# Get serotyped cases from model
Y_obs = df[['cluster','date']]
Y_obs[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = posterior_predictive['observed_data']['Y_obs'].values.reshape((n_months*n_clusters, n_serotypes))
Y_obs = Y_obs.set_index(['cluster','date'])

# Compute observed ratios
Y_obs['p_1'] = Y_obs['DENV_1']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])
Y_obs['p_2'] = Y_obs['DENV_2']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])
Y_obs['p_3'] = Y_obs['DENV_3']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])
Y_obs['p_4'] = Y_obs['DENV_4']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])

# Get total typed cases from model and data
N_typed = df[['date', 'cluster', 'DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
N_typed['N_typed_latent'] = N_typed[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].sum(axis=1)
N_typed['N_typed_latent'][N_typed['N_typed_latent'] == 0] = np.nan
N_typed = N_typed.set_index(['cluster','date'])['N_typed_latent']

# Get serotype fractions
## Mean
p_mean = df[['cluster','date']]
p_mean[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['p'].mean(dim=['chain','draw']).values.reshape((n_months*n_clusters, n_serotypes))
p_mean = p_mean.set_index(['cluster','date'])
## Lower
p_lower = df[['cluster','date']]
p_lower[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['p'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
p_lower = p_lower.set_index(['cluster','date'])
## Upper
p_upper = df[['cluster','date']]
p_upper[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['p'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
p_upper = p_upper.set_index(['cluster','date'])

# Get number of susceptibility slots per serotype
def get_susceptibles_serotype_time(S):
    """
    S: numpy array of shape (n_times, n_clusters, state_idx)

    Returns
    -------
    numpy array of shape (n_times, n_clusters, n_serotypes)
    """

    S1 = np.sum(S[:, :, [0, 2, 3, 4, 8, 9, 10, 11]], axis=2)
    S2 = np.sum(S[:, :, [0, 1, 3, 4, 6, 7, 10, 12]], axis=2)
    S3 = np.sum(S[:, :, [0, 1, 2, 4, 5, 7, 9, 13]], axis=2)
    S4 = np.sum(S[:, :, [0, 1, 2, 3, 5, 6, 8, 14]], axis=2)

    return np.stack([S1, S2, S3, S4], axis=-1)

## Mean
S_mean = df[['cluster','date']]
S_mean[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = get_susceptibles_serotype_time(trace['posterior']['S'].mean(dim=['chain','draw']).values).reshape((n_months*n_clusters, n_serotypes))
S_mean = S_mean.set_index(['cluster','date'])
## Lower
S_lower = df[['cluster','date']]
S_lower[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = get_susceptibles_serotype_time(trace['posterior']['S'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values).reshape((n_months*n_clusters, n_serotypes))
S_lower = S_lower.set_index(['cluster','date'])
## Upper
S_upper = df[['cluster','date']]
S_upper[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = get_susceptibles_serotype_time(trace['posterior']['S'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values).reshape((n_months*n_clusters, n_serotypes))
S_upper = S_upper.set_index(['cluster','date'])

# Get fraction of population per infection degree
def get_susceptibles_degree_time(S):
    """
    S: numpy array of shape (n_times, n_clusters, state_idx)

    Returns
    -------
    numpy array of shape (n_times, n_clusters, n_degrees)
    """

    S1 = S[:, :, 0]
    S2 = np.sum(S[:, :, [1, 2, 3, 4]], axis=2)
    S3 = np.sum(S[:, :, [5, 6, 7, 8, 9, 10]], axis=2)
    S4 = np.sum(S[:, :, [11, 12, 13, 14]], axis=2)
    S5 = S[:, :, 15]

    return np.stack([S1, S2, S3, S4, S5], axis=-1)

## Mean
d_mean = df[['cluster','date']]
d_mean[['0', '1', '2', '3', '4']] = get_susceptibles_degree_time(trace['posterior']['S'].mean(dim=['chain','draw']).values).reshape((n_months*n_clusters, 5))
d_mean = d_mean.set_index(['cluster','date'])
## Lower
d_lower = df[['cluster','date']]
d_lower[['0', '1', '2', '3', '4']] = get_susceptibles_degree_time(trace['posterior']['S'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values).reshape((n_months*n_clusters, 5))
d_lower = d_lower.set_index(['cluster','date'])
## Upper
d_upper = df[['cluster','date']]
d_upper[['0', '1', '2', '3', '4']] = get_susceptibles_degree_time(trace['posterior']['S'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values).reshape((n_months*n_clusters, 5))
d_upper = d_upper.set_index(['cluster','date'])


# Get total dengue cases
## Mean
D_mean = df[['cluster','date']]
D_mean['DENV_total'] = posterior_predictive['posterior_predictive']['D_obs'].mean(dim=['chain','draw']).values.reshape((n_months*n_clusters))
D_mean = D_mean.set_index(['cluster','date'])
## Lower
D_lower = df[['cluster','date']]
D_lower['DENV_total'] = posterior_predictive['posterior_predictive']['D_obs'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values.reshape((n_months*n_clusters))
D_lower = D_lower.set_index(['cluster','date'])
## Upper
D_upper = df[['cluster','date']]
D_upper['DENV_total'] = posterior_predictive['posterior_predictive']['D_obs'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values.reshape((n_months*n_clusters))
D_upper = D_upper.set_index(['cluster','date'])

# Get total FOI
## Mean
FOI_mean = df[['cluster','date']]
FOI_mean['FOI'] = trace['posterior']['lambda_t'].mean(dim=['chain','draw']).values.reshape((n_months*n_clusters))
FOI_mean = FOI_mean.set_index(['cluster','date'])
## Lower
FOI_lower = df[['cluster','date']]
FOI_lower['FOI'] = trace['posterior']['lambda_t'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values.reshape((n_months*n_clusters))
FOI_lower = FOI_lower.set_index(['cluster','date'])
## Upper
FOI_upper = df[['cluster','date']]
FOI_upper['FOI'] = trace['posterior']['lambda_t'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values.reshape((n_months*n_clusters))
FOI_upper = FOI_upper.set_index(['cluster','date'])

# Get z trajectory
## Mean
z_mean = df[['cluster','date']]
z_mean[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['z'].mean(dim=['chain','draw']).values.reshape((n_months*n_clusters, n_serotypes))
z_mean = z_mean.set_index(['cluster','date'])
## Lower
z_lower = df[['cluster','date']]
z_lower[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['z'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
z_lower = z_lower.set_index(['cluster','date'])
## Upper
z_upper = df[['cluster','date']]
z_upper[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['z'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
z_upper = z_upper.set_index(['cluster','date'])

# Get beta_f trajectory
## Mean
beta_f_mean = df[['cluster','date']]
beta_f_mean[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['beta_f'].mean(dim=['chain','draw']).values.reshape((n_months*n_clusters, n_serotypes))
beta_f_mean = beta_f_mean.set_index(['cluster','date'])
## Lower
beta_f_lower = df[['cluster','date']]
beta_f_lower[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['beta_f'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
beta_f_lower = beta_f_lower.set_index(['cluster','date'])
## Upper
beta_f_upper = df[['cluster','date']]
beta_f_upper[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['beta_f'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
beta_f_upper = beta_f_upper.set_index(['cluster','date'])

# Get residual trajectory
## Mean
eps_mean = df[['cluster','date']]
eps_mean[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['eps'].mean(dim=['chain','draw']).values.reshape((n_months*n_clusters, n_serotypes))
eps_mean = eps_mean.set_index(['cluster','date'])
## Lower
eps_lower = df[['cluster','date']]
eps_lower[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['eps'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
eps_lower = eps_lower.set_index(['cluster','date'])
## Upper
eps_upper = df[['cluster','date']]
eps_upper[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['eps'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values.reshape((n_months*n_clusters, n_serotypes))
eps_upper = eps_upper.set_index(['cluster','date'])

# Get overdispersion
## Mean
phi_mean = df[['cluster','date']]
phi_mean['phi'] = trace['posterior']['phi'].mean(dim=['chain','draw']).values.flatten()
phi_mean = phi_mean.set_index(['cluster','date'])
## Lower
phi_lower = df[['cluster','date']]
phi_lower['phi'] = trace['posterior']['phi'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values.flatten()
phi_lower = phi_lower.set_index(['cluster','date'])
## Upper
phi_upper = df[['cluster','date']]
phi_upper['phi'] = trace['posterior']['phi'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values.flatten()
phi_upper = phi_upper.set_index(['cluster','date'])

# Get timepoints
time = df['date'].unique()

###################
## Visualisation ##
###################

for cluster in df['cluster'].unique().tolist():
    
    # demography in start_year
    pop_start_year = demo[demo['cluster'] == cluster]['population'].values

    # Visualisation
    fig,ax=plt.subplots(nrows=13, sharex=True, figsize=(8.7, 11.3*1.25))

    # 1: Total dengue cases
    # Filter data for a single UF
    df_star_mean = D_mean.loc[cluster, ['DENV_total']]
    df_star_lower = D_lower.loc[cluster, ['DENV_total']]
    df_star_upper = D_upper.loc[cluster, ['DENV_total']]
    ax[0].plot(time, df[df['cluster']==cluster]['DENV_total'].values/pop_start_year*100, marker='o', markersize=2, linewidth=0.5, color='black')
    ax[0].plot(df_star_mean.index, df_star_mean[f'DENV_total']/pop_start_year*100, color='blue', alpha=1)
    ax[0].fill_between(df_star_mean.index, df_star_lower[f'DENV_total']/pop_start_year*100, df_star_upper[f'DENV_total']/pop_start_year*100, color='blue', alpha=0.2)
    ax[0].set_ylabel('Total DENV (%)', fontsize=7)
    #ax[0].set_ylim([-0.5,5])
    ax[0].set_title(f'Brasil (Cluster: {cluster})')

    # 2: total serotyped cases
    ax[1].plot(time, N_typed.loc[cluster, slice(None)].values, marker='o', markersize=2, linewidth=0.5, color='black')
    ax[1].set_ylim([0,300])
    ax[1].set_ylabel('Total serotyped (-)', fontsize=7)

    # 3: serotype fractions vs data
    colors = ['black', 'red', 'green', 'blue']
    for i in range(1,5):
        ax[1+i].plot(time, Y_obs.loc[(cluster, slice(None)), f'p_{i}'].values*100, marker='o', markersize=2, linewidth=1, color='black')
        ax[1+i].plot(time, p_mean.loc[cluster, f'DENV_{i}']*100, color=colors[i-1])
        ax[1+i].fill_between(time, p_lower.loc[cluster, f'DENV_{i}']*100, p_upper.loc[cluster, f'DENV_{i}']*100, alpha=0.2, color=colors[i-1])
        ax[1+i].set_ylabel(f'DENV {i} (%)', fontsize=7)
        ax[1+i].set_ylim([-3,103])

    # 4: FOI
    # Filter data for a single UF
    df_star_mean = FOI_mean.loc[cluster, ['FOI']]
    df_star_lower = FOI_lower.loc[cluster, ['FOI']]
    df_star_upper = FOI_upper.loc[cluster, ['FOI']]
    ax[6].axhline(np.mean(df_star_mean[f'FOI']*100), color='red', linewidth=0.5)
    ax[6].plot(df_star_mean.index, df_star_mean[f'FOI']*100, color='blue', alpha=1)
    ax[6].fill_between(df_star_mean.index, df_star_lower['FOI']*100, df_star_upper['FOI']*100, color='blue', alpha=0.2)
    ax[6].set_ylabel('Total FOI (%)', fontsize=7)
    ax[6].set_ylim([-0.5,30])

    # 5: susceptibility slots per serotype
    # Filter data for a single UF
    df_star_mean = S_mean.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_lower = S_lower.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_upper = S_upper.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    # Plot
    colors = ['black', 'red', 'green', 'blue']
    for i in range(1,5):
        ax[7].plot(df_star_mean.index, df_star_mean[f'DENV_{i}']/pop_start_year*100, label='1', color=colors[i-1], alpha=1)
        ax[7].fill_between(df_star_mean.index, df_star_lower[f'DENV_{i}']/pop_start_year*100, df_star_upper[f'DENV_{i}']/pop_start_year*100, label=f'{i}', color=colors[i-1], alpha=0.1)
        ax[7].set_ylim([-3,125])
    ax[7].set_ylabel('Susc. slots (%)', fontsize=7)

    # 6: modeled serotype fractions
    df_star = p_mean.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    ax[8].stackplot(
        df_star.index, [df_star['DENV_1']*100, df_star['DENV_2']*100, df_star['DENV_3']*100, df_star['DENV_4']*100],
        labels=['1', '2', '3', '4'],
        colors=['black', 'red', 'green', 'blue'],
        alpha=0.9
    )
    ax[8].legend(framealpha=1, loc=3, fontsize=7)
    ax[8].set_ylabel('Serotypes (%)', fontsize=7)

    # 7: susceptibles per degree of infection
    # Filter data for a single UF
    df_star_mean = d_mean.loc[cluster, ['0', '1', '2', '3', '4']]
    df_star_lower = d_lower.loc[cluster, ['0', '1', '2', '3', '4']]
    df_star_upper = d_upper.loc[cluster, ['0', '1', '2', '3', '4']]
    # Plot
    ax[9].stackplot(
        df_star_mean.index, [df_star_mean['0']/pop_start_year*100, df_star_mean['1']/pop_start_year*100, df_star_mean['2']/pop_start_year*100, df_star_mean['3']/pop_start_year*100, df_star_mean['4']/pop_start_year*100],
        labels=['0', '1', '2', '3', '4'],
        colors=['black', 'red', 'orange', 'yellow', 'green'],
        alpha=0.9
    )
    ax[9].set_ylabel('Susc. degree (%)', fontsize=7)

    # 8: beta * f_i
    # Filter data for a single UF
    df_star_mean = beta_f_mean.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_lower = beta_f_lower.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_upper = beta_f_upper.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    # Plot
    colors = ['black', 'red', 'green', 'blue']
    for i in range(1,5):
        ax[10].plot(df_star_mean.index, df_star_mean[f'DENV_{i}'], label='1', color=colors[i-1], alpha=1)
        ax[10].fill_between(df_star_mean.index, df_star_lower[f'DENV_{i}'], df_star_upper[f'DENV_{i}'], label=f'{i}', color=colors[i-1], alpha=0.1)
        ax[10].set_ylim([-0.2,0.2])
    ax[10].set_ylabel('beta * f_i (-)', fontsize=7)

    # 9: residual
    # Filter data for a single UF
    df_star_mean = eps_mean.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_lower = eps_lower.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_upper = eps_upper.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    # Plot
    colors = ['black', 'red', 'green', 'blue']
    for i in range(1,5):
        ax[11].plot(df_star_mean.index, df_star_mean[f'DENV_{i}'], label='1', color=colors[i-1], alpha=1)
        ax[11].fill_between(df_star_mean.index, df_star_lower[f'DENV_{i}'], df_star_upper[f'DENV_{i}'], label=f'{i}', color=colors[i-1], alpha=0.1)
        ax[11].set_ylim([-0.2,0.2])
    ax[11].set_ylabel('Residual (-)', fontsize=7)

    # 10: z
    # Filter data for a single UF
    df_star_mean = z_mean.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_lower = z_lower.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    df_star_upper = z_upper.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    # Plot
    colors = ['black', 'red', 'green', 'blue']
    for i in range(1,5):
        ax[12].plot(df_star_mean.index, df_star_mean[f'DENV_{i}'], label='1', color=colors[i-1], alpha=1)
        ax[12].fill_between(df_star_mean.index, df_star_lower[f'DENV_{i}'], df_star_upper[f'DENV_{i}'], label=f'{i}', color=colors[i-1], alpha=0.1)
    ax[12].set_ylabel('z (-)', fontsize=7)


    os.makedirs(f'{output_folder}/fig/posterior_predictive', exist_ok=True)
    plt.savefig(f'{output_folder}/fig/posterior_predictive/{cluster}_total_serotyped.pdf')
    #plt.show()
    plt.close()