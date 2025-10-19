import os
import arviz
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# analysis startdate
startdate = datetime(1900,1,1)

# helper function for argument parsing
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# arguments are used to find the result
# How to run: python visualise-fit.py -date 2025-08-27 -ID test -p 2 -distance_matrix False
parser = argparse.ArgumentParser()
parser.add_argument("-ID", type=str, help="Identifier of the pipeline run.")
parser.add_argument("-spatial_aggregation", type=str, help="Spatial aggregation clustering was performed on.")
parser.add_argument("-p", type=int, help="Order of AR(p) process.", default=1)
parser.add_argument("-q", type=int, help="Order of MA(q) process.", default=1)
args = parser.parse_args()

# assign to desired variables
spatial_aggregation = args.spatial_aggregation
ID = args.ID
p = args.p
q = args.q

# pipeline output folder
abs_dir = os.path.dirname(__file__) # make sure all referenced paths are relative to the lcoation of this file and not the terminal's pwd
output_folder = os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/bayesian-imputation-model_output/ARMA({p},{q})/')
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

##################################
## Preparing the incidence data ##
##################################

# Load mapping
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))

# Load clusters
clusters = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/clusters/clusters_{spatial_aggregation}.csv'))
region = clusters.columns.to_list()[0]

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
# save result
output_mun.to_parquet(f'{output_folder}/DENV-serotypes-imputed_1996-2025_monthly.parquet', compression='brotli')



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
        
    # Visualisation
    fig,ax=plt.subplots(nrows=7, sharex=True, figsize=(8.7, 11.3))

    # Step 1: total serotyped cases
    ax[0].plot(time, N_typed.loc[cluster, slice(None)].values, marker='o', markersize=2, linewidth=0.5, color='black')
    ax[0].set_ylim([0,200])
    ax[0].set_ylabel('Total serotyped (-)')
    ax[0].set_title(f'Brasil (Cluster: {cluster})')

    # Step 2: serotype fractions vs data
    ## DENV 1
    ax[1].plot(time, Y_obs.loc[(cluster, slice(None)), 'p_1'].values*100, marker='o', markersize=2, linewidth=1, color='black')
    ax[1].plot(time, p_mean.loc[cluster, 'DENV_1']*100, color='red')
    ax[1].fill_between(time, p_lower.loc[cluster, 'DENV_1']*100, p_upper.loc[cluster, 'DENV_1']*100, alpha=0.2, color='red')
    ax[1].set_ylabel('DENV 1 (%)')
    ax[1].set_ylim([-3,103])
    ## DENV 2
    ax[2].plot(time, Y_obs.loc[(cluster, slice(None)), 'p_2'].values*100, marker='o', markersize=2, linewidth=1, color='black')
    ax[2].plot(time, p_mean.loc[cluster, 'DENV_2']*100, color='red')
    ax[2].fill_between(time, p_lower.loc[cluster, 'DENV_2']*100, p_upper.loc[cluster, 'DENV_2']*100, alpha=0.2, color='red')
    ax[2].set_ylabel('DENV 2 (%)')
    ax[2].set_ylim([-3,103])
    ## DENV 3
    ax[3].plot(time, Y_obs.loc[(cluster, slice(None)), 'p_3'].values*100, marker='o', markersize=2, linewidth=1, color='black')
    ax[3].plot(time, p_mean.loc[cluster, 'DENV_3']*100, color='red')
    ax[3].fill_between(time, p_lower.loc[cluster, 'DENV_3']*100, p_upper.loc[cluster, 'DENV_3']*100, alpha=0.2, color='red')
    ax[3].set_ylabel('DENV 3 (%)')
    ax[3].set_ylim([-3,103])
    ## DENV 4
    ax[4].plot(time, Y_obs.loc[(cluster, slice(None)), 'p_4'].values*100, marker='o', markersize=2, linewidth=1, color='black', label='data')
    ax[4].plot(time, p_mean.loc[cluster, 'DENV_4']*100, color='red', label='model')
    ax[4].fill_between(time, p_lower.loc[cluster, 'DENV_4']*100, p_upper.loc[cluster, 'DENV_4']*100, alpha=0.2, color='red')
    ax[4].set_ylabel('DENV 4 (%)')
    ax[4].set_ylim([-3,103])
    ax[4].legend(framealpha=1)

    # Step 3: modeled serotype fractions
    # Filter data for a single UF
    df_star = p_mean.loc[cluster, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
    # Plot manually
    ax[5].stackplot(
        df_star.index, [df_star['DENV_1']*100, df_star['DENV_2']*100, df_star['DENV_3']*100, df_star['DENV_4']*100],
        labels=['1', '2', '3', '4'],
        colors=['black', 'red', 'green', 'blue'],
        alpha=0.9
    )
    ax[5].legend(framealpha=1)
    ax[5].set_ylabel('Serotypes (%)')

    # Step 4: modeled overdispersion factor
    ax[6].plot(time, phi_mean.loc[cluster, 'phi'], color='red')
    ax[6].fill_between(time, phi_lower.loc[cluster, 'phi'], phi_upper.loc[cluster, 'phi'], alpha=0.2, color='red')
    ax[6].set_ylabel(r'$\phi(t)$ (-)')
    #ax[6].set_ylim([-3,103])

    os.makedirs(f'{output_folder}/fig/posterior_predictive', exist_ok=True)
    plt.savefig(f'{output_folder}/fig/posterior_predictive/{cluster}_total_serotyped.pdf')
    #plt.show()
    plt.close()