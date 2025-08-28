import os
import arviz
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# helper function for argument parsing
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# arguments are used to find the result
# How to run: python visualise-fit.py -date 2025-08-27 -ID test -p 2 -distance_matrix False -CAR_per_lag False
parser = argparse.ArgumentParser()
parser.add_argument("-state", type=str, help="Abbreviation of brazilian federative unit.")
parser.add_argument("-date", type=str, help="Date experiment was run.")
parser.add_argument("-ID", type=str, help="Sampler output name.")
parser.add_argument("-p", type=int, help="Order of AR(p) process.", default=1)
parser.add_argument("-distance_matrix", type=str_to_bool, help="Use distance matrix versus adjacency matrix.", default=False)
parser.add_argument("-CAR_per_lag", type=str_to_bool, help="Use one spatial innovation process per AR lag versus one spatial innovation overall.", default=False)
args = parser.parse_args()

# assign to desired variables
state = args.state
date = args.date
ID = args.ID
p = args.p
distance_matrix = args.distance_matrix
CAR_per_lag = args.CAR_per_lag

# Make folder structure
output_folder=f'../../data/interim/bayesian-imputation-model_output/AR({p})/distance_matrix-{distance_matrix}/CARperlag-{CAR_per_lag}/{ID}_{date}' # Path to backend
# check if samples folder exists, if not, make it
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

# Load the posterior predictive check
ppc = arviz.from_netcdf(f"{output_folder}/ppc.nc")

##################################
## Preparing the incidence data ##
##################################

# Fetch incidence data
df = pd.read_csv('../../data/interim/datasus_DENV-linelist/DENV-serotypes_1996-2025_monthly.csv', parse_dates=['date'])

# 1. Check if all columns are present
sero_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4"]
required_cols = ["UF", "date", "DENV_total"] + sero_cols
assert all(col in df.columns for col in required_cols)

# 2. Sort for safety
df = df.sort_values(["date", "UF"]).reset_index(drop=True)

df = df[df['date'] > datetime(1999,1,1)]

# 3. Factorize states and time
df["state_idx"], _ = pd.factorize(df["UF"])
df["month_idx"], _ = pd.factorize(df["date"])

# 4. Fill NaNs in a principled way
def fill_serotypes(row):
    sero = row[sero_cols]
    if sero.notna().any():
        # If at least one serotype is observed, treat missing ones as 0
        for col in sero_cols:
            if pd.isna(row[col]):
                row[col] = 0.0
    return row
df = df.apply(fill_serotypes, axis=1)

# 5. Compute N_typed
df["N_typed"] = df[sero_cols].sum(axis=1, skipna=False)                                         # if serotypes available --> sum them
df.loc[df[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].isna().all(axis=1), 'N_typed'] = np.nan      # if all serotypes are Nan --> N_typed = 0 --> Wait, I don't think this is appropriate.

# 6. Compute delta (typing fraction)
df["delta"] = df["N_typed"] / df["DENV_total"]
df['delta'] = df['delta'].where(df['N_typed'] > 0, np.nan) # When N_typed == 0, we don't know delta — mark as missing
df["delta"] = df["delta"].clip(lower=1e-12, upper=1 - 1e-12)

# 7. Compute year index
df["year"] = pd.to_datetime(df["date"]).dt.year
df["year_idx"] = df["year"] - df["year"].min()

# 8. Compute year-state index
df["state_year_idx"] = df["state_idx"].astype(str) + "_" + df["year_idx"].astype(str)
df["state_year_idx"], state_year_labels = pd.factorize(df["state_year_idx"])

# 9. Build PyMC arrays

# --- For Beta model (typing fraction, always available) ---
delta_obs = df["delta"].to_numpy().astype(float)
N_total = df["DENV_total"].to_numpy().astype(int)

# --- For Multinomial model (subtypes, only when typed) ---
Y_multinomial = df[sero_cols].to_numpy().astype(int)
N_typed = df["N_typed"].to_numpy().astype(int)


################################
## Output the imputed dataset ##
################################

output = df[['date', 'UF', 'DENV_1', 'DENV_2', 'DENV_3', 'DENV_4', 'DENV_total']]
output[['p_1', 'p_2', 'p_3', 'p_4']] = trace['posterior']['p'].mean(dim=['chain','draw']).values
output.to_csv(f'{output_folder}/DENV-serotypes-imputed_1996-2025_monthly.csv')


################################################
## Prepare the data used in the visualisation ##
################################################

# Get serotyped cases from model
Y_obs = df[['UF','date']]
Y_obs[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = ppc['observed_data']['Y_obs'].values
Y_obs = Y_obs.set_index(['UF','date'])

# Compute observed ratios
Y_obs['p_1'] = Y_obs['DENV_1']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])
Y_obs['p_2'] = Y_obs['DENV_2']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])
Y_obs['p_3'] = Y_obs['DENV_3']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])
Y_obs['p_4'] = Y_obs['DENV_4']/(Y_obs['DENV_1'] + Y_obs['DENV_2'] + Y_obs['DENV_3'] + Y_obs['DENV_4'])

# Get total typed cases from model and data
N_typed = df[['date', 'UF', 'DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
N_typed['N_typed_latent'] = N_typed[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].sum(axis=1)
N_typed['N_typed_latent'][N_typed['N_typed_latent'] == 0] = np.nan
N_typed = N_typed.set_index(['UF','date'])['N_typed_latent']
print(N_typed)

# Get serotype fractions
## Mean
p_mean = df[['UF','date']]
p_mean[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['p'].mean(dim=['chain','draw']).values
p_mean = p_mean.set_index(['UF','date'])
## Lower
p_lower = df[['UF','date']]
p_lower[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['p'].quantile(dim=['chain','draw'], q=(100-confidence)/2/100).values
p_lower = p_lower.set_index(['UF','date'])
## Upper
p_upper = df[['UF','date']]
p_upper[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']] = trace['posterior']['p'].quantile(dim=['chain','draw'], q=1-(100-confidence)/2/100).values
p_upper = p_upper.set_index(['UF','date'])

# Get timepoints
time = df['date'].unique()


###################
## Visualisation ##
###################

# Visualisation
fig,ax=plt.subplots(nrows=6, sharex=True, figsize=(8.7, 11.3))

# Step 1: total serotyped cases
ax[0].scatter(time, N_typed.loc[state, slice(None)].values, marker='o', s=2, color='black')
ax[0].set_ylim([0,500])
ax[0].set_ylabel('Total serotyped (-)')
ax[0].set_title(f'Brasil (State: {state})')

# Step 2: serotype fractions vs data
## DENV 1
ax[1].plot(time, Y_obs.loc[(state, slice(None)), 'p_1'].values*100, marker='o', markersize=2, linewidth=1, color='black')
ax[1].plot(time, p_mean.loc[state, 'DENV_1']*100, color='red')
ax[1].fill_between(time, p_lower.loc[state, 'DENV_1']*100, p_upper.loc[state, 'DENV_1']*100, alpha=0.2, color='red')
ax[1].set_ylabel('DENV 1 (%)')
## DENV 2
ax[2].plot(time, Y_obs.loc[(state, slice(None)), 'p_2'].values*100, marker='o', markersize=2, linewidth=1, color='black')
ax[2].plot(time, p_mean.loc[state, 'DENV_2']*100, color='red')
ax[2].fill_between(time, p_lower.loc[state, 'DENV_2']*100, p_upper.loc[state, 'DENV_2']*100, alpha=0.2, color='red')
ax[2].set_ylabel('DENV 2 (%)')
## DENV 3
ax[3].plot(time, Y_obs.loc[(state, slice(None)), 'p_3'].values*100, marker='o', markersize=2, linewidth=1, color='black')
ax[3].plot(time, p_mean.loc[state, 'DENV_3']*100, color='red')
ax[3].fill_between(time, p_lower.loc[state, 'DENV_3']*100, p_upper.loc[state, 'DENV_3']*100, alpha=0.2, color='red')
ax[3].set_ylabel('DENV 3 (%)')
## DENV 4
ax[4].plot(time, Y_obs.loc[(state, slice(None)), 'p_4'].values*100, marker='o', markersize=2, linewidth=1, color='black', label='data')
ax[4].plot(time, p_mean.loc[state, 'DENV_4']*100, color='red', label='model')
ax[4].fill_between(time, p_lower.loc[state, 'DENV_4']*100, p_upper.loc[state, 'DENV_4']*100, alpha=0.2, color='red')
ax[4].set_ylabel('DENV 4 (%)')
ax[4].legend(framealpha=1)

# Step 3: modeled serotype fractions
# Filter data for a single UF
df_star = p_mean.loc[state, ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]
# Plot manually
ax[5].stackplot(
    df_star.index, [df_star['DENV_1']*100, df_star['DENV_2']*100, df_star['DENV_3']*100, df_star['DENV_4']*100],
    labels=['1', '2', '3', '4'],
    colors=['black', 'red', 'green', 'blue'],
    alpha=0.9
)
ax[5].legend(framealpha=1)
ax[5].set_ylabel('Serotypes (%)')

plt.savefig(f'{output_folder}/{state}_total_serotyped.pdf')
#plt.show()
plt.close()