n_chains = 4
nuts_progress_bar = True

import os
import time
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from DENV_transmission_model.jax_forward_sim_model import DengueModel
from DENV_transmission_model.numpyro_models import single_cluster_training_model, single_cluster_training_model_RV_dims
from DENV_transmission_model.numpyro_utils import get_incidence_data, get_seroprevalence_data

# jax and numpyro
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro
numpyro.set_host_device_count(n_chains)
from numpyro.infer import NUTS, MCMC, init_to_value, init_to_median
import arviz


abs_dir = os.path.dirname(__file__)


# model settings
## what cluster of municipalities to simulate
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))
muni_clusters = mapping.loc[mapping['CD_RGINT'] == 2601, ('CD_MUN', 'CD_RGINT')]
n_munis = len(muni_clusters['CD_RGINT'].unique())
## what age groups to simulate
age_groups = [(0, 1), (1, 6), (6, 12), (12, 18), (18, 25), (25,35), (35,55), (55,120)]
n_age_groups = len(age_groups)
n_serotypes = 4
## when to start the simulation
start_sim = datetime(1998, 9, 1)
end_sim = datetime(2002, 1, 1)
simulation_dates = pd.date_range(start=start_sim, end=end_sim, freq="ME").to_pydatetime().tolist()

## when to introduce DENV 3 and DENV 4
intro_dates_DENV3 = [datetime(2000, 1, 1),]
intro_dates_DENV4 =[datetime(2007, 1, 1),]


# Initialise the transmission model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

transmission_model = DengueModel(age_groups, muni_clusters, start_sim, intro_dates_DENV3, intro_dates_DENV4)
n_months = len(transmission_model.months)


# Fetch the fitting data
# ~~~~~~~~~~~~~~~~~~~~~~

## total incidence
data_dir = os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/DENV-1998_2026-month-mun-no_diagnostics.parquet')
data_dates, total_incidence, serotype_incidence = get_incidence_data(data_dir, muni_clusters, start_sim)
serotype_incidence = jnp.nan_to_num(serotype_incidence, nan=0.0)    # dirichletMultinomial doesn't work with NA
## seroprevalence
data_dir = os.path.join(abs_dir, '../../data/interim/seroprevalence/seroprevalence.csv')
seroprevalence = get_seroprevalence_data(data_dir, age_groups, muni_clusters)


# visualise the seroprevalence data
## one panel per survey year, aligned in a single row; experiments from the same year share a panel
for cluster, seroprev_df in seroprevalence.items():
    ## group the experiments per survey year
    experiments_by_year = {}
    for experiment in seroprev_df.columns:
        year, study_id = experiment.split('_')
        experiments_by_year.setdefault(year, []).append(experiment)
    ## one panel per year
    fig, axes = plt.subplots(nrows=1, ncols=len(experiments_by_year), squeeze=False, figsize=(8.3,11.7/3))
    for ax, (year, experiments) in zip(axes[0], sorted(experiments_by_year.items())):
        seroprev_df[experiments].plot.bar(ax=ax, legend=False)
        ax.set_title(f"{year}")
        ax.set_xlabel('age group (years)')
        ax.set_ylabel('seroprevalence (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    plt.close()


# Find the model's MAP
# ~~~~~~~~~~~~~~~~~~~~


# Sample numpyro model
# ~~~~~~~~~~~~~~~~~~~~

# fit the numpyro model with MCMC
start_dt = datetime.now()
start_time = time.time()

print(f"\nstarting the NUTS sampler at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ..")

rng_key = jax.random.PRNGKey(int(time.time()))
rng_key, rng_predict = jax.random.split(rng_key)

# construct its arguments
model_kwargs = dict(
    transmission_model=transmission_model,
    simulation_dates=simulation_dates,
    total_incidence=total_incidence,
    serotype_incidence=serotype_incidence,
    data_dates=data_dates
)

# initialise the NUTS sampler
kernel = NUTS(
    single_cluster_training_model,
    adapt_step_size=True,
    max_tree_depth=10,
    target_accept_prob=0.8,
    init_strategy = init_to_median,     # change to MAP
)

mcmc = MCMC(
    kernel,
    num_warmup=500,
    num_samples=500,
    num_chains=4,
    chain_method="parallel",
    progress_bar=nuts_progress_bar,
)

mcmc.run(
    rng_key,
    **model_kwargs,
    extra_fields=["potential_energy", "adapt_state.step_size", "diverging"]
)

# Chain collection prevents jax asynchronous dispatch from weirdly sequencing printouts
time.sleep(1)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), mcmc.get_samples())

# Record the end timestamp and compute elapsed time
end_dt = datetime.now()
elapsed_seconds = time.time() - start_time
elapsed_formatted = str(timedelta(seconds=int(elapsed_seconds)))

print(f"..and finished sampling at: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
print(f"total elapsed time: {elapsed_formatted}\n")
print(f"there were {int(jnp.sum(mcmc.get_extra_fields()["diverging"]))} divergent transitions\n")


print(mcmc.get_extra_fields()["potential_energy"])