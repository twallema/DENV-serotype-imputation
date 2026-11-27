n_chains = 1
nuts_progress_bar = True
find_new_map = False
output_folder = ''


import os
import time
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from DENV_transmission_model.jax_forward_sim_model import DengueModel
from DENV_transmission_model.numpyro_models import single_cluster_training_model, single_cluster_training_model_RV_dims, calculate_simend_data_index
from DENV_transmission_model.numpyro_utils import get_incidence_data, get_mean_case_age, get_seroprevalence_data, find_map

# jax and numpyro
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro
numpyro.set_host_device_count(n_chains)
from numpyro.infer import NUTS, MCMC, Predictive, init_to_value
import arviz


abs_dir = os.path.dirname(__file__)


# model settings
## what cluster of municipalities to simulate
cluster = 3301
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))
muni_clusters = mapping.loc[mapping['CD_RGINT'] == cluster, ('CD_MUN', 'CD_RGINT')]    # TODO: switch to clustering
## what age groups to simulate
age_groups = [(0, 1), (1, 6), (6, 12), (12, 18), (18, 25), (25,35), (35,55), (55,120)]
age_groups_labels = [f"[{start}, {end})" for start, end in age_groups]
n_age_groups = len(age_groups)
n_serotypes = 4
## when to start the simulation
start_sim = datetime(1998, 9, 1)    
end_sim = datetime(2016, 1, 1)
simulation_dates = pd.date_range(start=start_sim, end=end_sim, freq="ME").to_pydatetime().tolist()

## when to introduce DENV 3 and DENV 4
intro_dates_DENV3 = [datetime(2000, 7, 1),] # Rio: 2001, February; Recife: 1999, March
intro_dates_DENV4 =[datetime(2011, 1, 1),]  # Rio: 2011, January; Recife: 2006, January

# get a previously obtained MAP estimate
if not find_new_map:
    with jnp.load(os.path.join(output_folder, "map_estimate.npz")) as archive:
        map_params = {key: jnp.asarray(archive[key]) for key in archive.files}
else:
    map_params = None


# Initialise the transmission model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

transmission_model = DengueModel(age_groups, muni_clusters, start_sim, intro_dates_DENV3, intro_dates_DENV4)
n_months = len(transmission_model.months)


# Fetch the fitting data
# ~~~~~~~~~~~~~~~~~~~~~~

## total incidence
data_dir = os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/DENV-1998_2026-month-mun-no_diagnostics.parquet')
data_dates, total_incidence, serotype_incidence = get_incidence_data(data_dir, muni_clusters, start_sim)
serotype_incidence = jnp.nan_to_num(serotype_incidence, nan=0.0)    # dirichletMultinomial --> ignores all zero rows in likelihood
total_incidence = jnp.nan_to_num(total_incidence, nan=1.0)          # fix missing values at start in Rio
## mean case age per year
data_dir = os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/DENV-1999_2026-month-mun-age_group.parquet')
_, mean_case_age_year, total_incidence_year = get_mean_case_age(data_dir, muni_clusters, start_sim, age_groups)
## seroprevalence
data_dir = os.path.join(abs_dir, '../../data/interim/seroprevalence/seroprevalence.csv')
seroprevalence_pandas, seroprevalence_array = \
    get_seroprevalence_data(data_dir, age_groups, muni_clusters, data_dates)

# visualise the serotyped fractions
## used to set introduction dates of DENV3 and DENV4
## most likely: lots of cross protection at sim start due to large DENV-2 outbreak
fig,ax=plt.subplots(nrows=2)
ax[0].plot(data_dates, total_incidence, marker='o', markerfacecolor=None, markersize=2)
ax[1].stackplot(data_dates, jnp.transpose(jnp.squeeze(transmission_model.seeding)))
plt.close()

# visualise the seroprevalence data
## one panel per survey year, aligned in a single row; experiments from the same year share a panel
for cluster, seroprev_df in seroprevalence_pandas.items():
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

# construct numpyro model parameters
model_kwargs = dict(
    transmission_model=transmission_model,
    simulation_dates=simulation_dates,
    total_incidence=total_incidence,
    serotype_incidence=serotype_incidence,
    seroprevalence=seroprevalence_array,
    data_dates=data_dates
)

print('pre-optimizing MAP\n')
print('(iter, score)', flush=True)

# run optimiser
map_params = find_map(single_cluster_training_model, model_kwargs, 2500, map_params=map_params)

# Save dictionary to a file
if find_new_map:
    jnp.savez(os.path.join(output_folder, "map_estimate.npz"), **map_params)

print(map_params['phi'])
print(map_params['f_P_raw'])
print(map_params['f_P2_raw'])
print(map_params['foi_year'])
print(jnp.exp(map_params['log_or_34']))
print(jnp.exp(map_params['log_or_serotype']))
print(jnp.exp(map_params['log_or_hom']))

# visualise the result
end_sim_data_index = calculate_simend_data_index(simulation_dates, data_dates)

## likelihood
serotype_incidence_rel = serotype_incidence / jnp.sum(serotype_incidence, axis=-1)[:, None]
fig,ax=plt.subplots(nrows=7, figsize=(8.3, 11.7/3*2))
ax[0].plot(data_dates[:end_sim_data_index+1], total_incidence[:end_sim_data_index+1], marker='o', markerfacecolor=None, markersize=2, color='black')
ax[0].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['dI_obs'], axis=-1)), color='red')
ax[0].set_ylabel("DENV inc. (-)", fontsize=8)
ax[1].plot(data_dates[:end_sim_data_index+1], jnp.sum(serotype_incidence[:end_sim_data_index+1], axis=-1), marker='o', markerfacecolor=None, markersize=2, color='black')
ax[1].set_ylabel("Serotyped (-)", fontsize=8)
for i in range(4):
    ax[i+2].plot(data_dates[:end_sim_data_index+1], 100*serotype_incidence_rel[:end_sim_data_index+1, 0, i], marker='o', markerfacecolor=None, markersize=2, color='black')
    ax[i+2].plot(simulation_dates, 100*jnp.squeeze(map_params['p_true'])[:, i], color='red')
    ax[i+2].set_ylabel(f"DENV {i+1} (%)", fontsize=8)
ax[i+3].stackplot(simulation_dates, 100*jnp.transpose(jnp.squeeze(map_params['p_true'])), colors=['black', 'red', 'green', 'blue'], labels=['DENV1', 'DENV2', 'DENV3', 'DENV4'])
ax[i+3].legend(fontsize=6, loc=1, framealpha=1)
ax[i+3].set_ylabel("Serotypes (%)", fontsize=8)
plt.show()
plt.close()

## Seroprevalence data versus simulation
### Find timesteps that are not nan
valid_mask = jnp.any(~jnp.isnan(jnp.squeeze(seroprevalence_array[:end_sim_data_index+1])), axis=1)
indices = jnp.nonzero(valid_mask)[0]
### Initialise figure with correct ncols
if len(indices) > 0:
    fig,axes = plt.subplots(ncols = len(indices), squeeze=False, sharey=True)
    ### Pull results and visualise
    axes = axes.ravel()
    for idx, ax in zip(indices, axes):
        data = 100*jnp.squeeze(seroprevalence_array[:end_sim_data_index+1])[idx]
        model = 100*jnp.squeeze(map_params['seroprev_model'][:end_sim_data_index+1])[idx]
        ax.bar(age_groups_labels, data, color='black')
        ax.plot(age_groups_labels, model, marker = 'x', color='red')
        ax.set_ylim([0,100])
        ax.set_title(simulation_dates[idx].strftime('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=30)
        ax.set_xlabel('Age group')    
    axes[0].set_ylabel('Seroprevalence (%)')
    plt.tight_layout()
    plt.show()
    plt.close()

## mean case age per year
import numpy as np
### sim output to years
data = jnp.squeeze(map_params['dI_obs'])
years = pd.to_datetime(simulation_dates).year
unique_years, year_idx = np.unique(years, return_inverse=True)
yearly_sum = jnp.zeros((len(unique_years), data.shape[1]), dtype=data.dtype).at[year_idx].add(data)
### to mean case age
midpoints = jnp.array([(low + high) / 2.0 for low, high in age_groups])
total_age = jnp.sum(yearly_sum * midpoints, axis=1)
total_cases = jnp.sum(yearly_sum, axis=1)
mean_case_age_year_model = total_age / total_cases

fig,ax = plt.subplots()
ax.plot(unique_years, mean_case_age_year_model, color='red')
ax.plot(unique_years, mean_case_age_year[:len(mean_case_age_year_model)], marker='o', color='black')
plt.show()
plt.close()

## latent states
fig,ax=plt.subplots(nrows=5, figsize=(8.3, 11.7/3*2))
### Transmission
ax[0].plot(data_dates[:end_sim_data_index], jnp.squeeze(map_params['beta_monthly_season'])[:end_sim_data_index], color='black', label='seasonal', linewidth=1)
ax[0].plot(data_dates[:end_sim_data_index], jnp.squeeze(map_params['beta_monthly_shock'])[:end_sim_data_index], color='black', linestyle = '-.', label='shock', linewidth=1)
ax[0].plot(data_dates[:end_sim_data_index], jnp.squeeze(map_params['beta_monthly'])[:end_sim_data_index], color='red', label='total', linewidth=3)
ax[0].set_ylabel(r'$\beta(t)$', fontsize=8)
ax[0].legend(fontsize=8, loc=1, framealpha=1)
### Homologous reinfection
from cycler import cycler
ax[1].set_prop_cycle(cycler('color', ['black', 'red', 'green', 'blue']))
ax[1].plot(data_dates[:end_sim_data_index], 100*jnp.squeeze(map_params['p_hom_monthly'])[:end_sim_data_index], label=['DENV1', 'DENV2', 'DENV3', 'DENV4'])
ax[1].set_ylabel(r'$p_{hom,i}(t)$ (%)', fontsize=8)
ax[1].legend(fontsize=8, loc=1, framealpha=1)
### Susceptibility slots per serotype
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3, 5)))[:, 0], color='black', label='DENV1')
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3, 5)))[:, 1], color='red', label='DENV2')
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3, 5)))[:, 2], color='green', label='DENV3')
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3, 5)))[:, 3], color='blue', label='DENV4')
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3)))[:, 0, 0], color='black', label='DENV1', linestyle='--')
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3)))[:, 1, 0], color='red', label='DENV2', linestyle='--')
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3)))[:, 2, 0], color='green', label='DENV3', linestyle='--')
ax[2].plot(simulation_dates, jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 3)))[:, 3, 0], color='blue', label='DENV4', linestyle='--')
ax[2].set_ylabel('Susceptibility slots (-)', fontsize=8)
### Susceptibles by number of previous heterologous infections
ax[3].stackplot(simulation_dates, jnp.transpose(jnp.squeeze(jnp.sum(map_params['S_expanded'], axis=(2, 4, 5)))), colors=['black', 'red', 'orange', 'yellow', 'green'], labels=['0', '1', '2', '3', '4'])
ax[3].legend(fontsize=8, loc=1, framealpha=1)
ax[3].set_ylabel('Susceptibles (-)', fontsize=8)
### FOI
ax[4].plot(simulation_dates, jnp.squeeze(map_params['lambda_t'])*100, color='black')
ax[4].set_ylabel('FOI (%)', fontsize=8)
plt.show()
plt.close()


# Sample numpyro model
# ~~~~~~~~~~~~~~~~~~~~

# fit the numpyro model with MCMC
start_dt = datetime.now()
start_time = time.time()

print(f"\nstarting the NUTS sampler at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ..")

rng_key = jax.random.PRNGKey(int(time.time()))
rng_key, rng_predict = jax.random.split(rng_key)

# initialise the NUTS sampler
kernel = NUTS(
    single_cluster_training_model,
    adapt_step_size=True,
    max_tree_depth=10,
    target_accept_prob=0.8,
    init_strategy = init_to_value(values=map_params),
)

mcmc = MCMC(
    kernel,
    num_warmup=5,
    num_samples=5,
    num_chains=n_chains,
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

# compute the likelihood of the last sample per likelihood site
from numpyro.infer import log_likelihood
posterior_samples = mcmc.get_samples()
ll_dict = log_likelihood(single_cluster_training_model, posterior_samples, **model_kwargs)

print("Untempered likelihood of last sample per likelihood site:")
print(f"Y_obs: {jnp.sum(ll_dict['Y_obs'][-1, ...])}")
print(f"D_obs:  {jnp.sum(ll_dict['D_obs'][-1, ...])}")
print(f"seroprev_obs: {jnp.sum(ll_dict['seroprev_obs'][-1, ...])}\n")


# Visualise traces
# ~~~~~~~~~~~~~~~~

# define coordinates
coords = {
    'cluster': [cluster,],
    'age_group': age_groups_labels,
    'data_date': data_dates,
    'simulation_date': simulation_dates,
    'serotype': [1, 2, 3, 4],
    'n_prior_het_infections_pi_d': [0, 1, 2],
    'n_prior_het_infections': [0, 1, 2, 3, 4],
    'current_het_infection': [1, 2, 3, 4],
    'infection_type': ['hom', 'het']
}

# convert to arviz
trace = arviz.from_numpyro(mcmc, coords=coords, dims=single_cluster_training_model_RV_dims)

# save traces to a netcdf
trace.to_netcdf(os.path.join(output_folder, "trace.nc"))

# Generate traceplots
variables2plot = ['pi_mono2_raw', 'f_P_raw', 'f_P2_raw', 'I_estim', 'foi_year', 'pi_d', 'f1_coeff', 'f2_coeff', 'mu_beta', 'A_beta', 'phi_beta', 'sigma_ar_beta',
                  'kappa0_logit', 'omega', 'log_or_34', 'log_or_serotype', 'log_or_hom', 'phi']

# Save original traces
os.makedirs(os.path.join(output_folder, 'traces'), exist_ok=True)
for var in variables2plot:
    arviz.plot_trace_dist(trace, var_names=[var], compact=True, combined=True, kind='kde') 
    plt.savefig(os.path.join(output_folder, f'traces/trace-{var}.pdf'))
    plt.close()


# Sample posterior predictive
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\nsampling posterior predictive\n')

predictive = Predictive(
    single_cluster_training_model,
    posterior_samples=mcmc.get_samples(),
    )

model_kwargs.update({'posterior_predictive': int(jnp.max(jnp.sum(serotype_incidence[:end_sim_data_index+1], axis=-1)))})    # maximum counts in DirichletMultinomial needs to come from outside the numpyro model; quircky

posterior_predictive = predictive(rng_predict, **model_kwargs)

# convert to arviz
posterior_predictive = arviz.from_numpyro(mcmc, posterior_predictive=posterior_predictive, coords=coords, dims=single_cluster_training_model_RV_dims)

# save posterior predictive
posterior_predictive.to_netcdf(os.path.join(output_folder, "posterior_predictive.nc"))


# Visualise goodness-of-fit
# ~~~~~~~~~~~~~~~~~~~~~~~~~

trace = arviz.from_netcdf('trace.nc')
pp = arviz.from_netcdf('posterior_predictive.nc')

## likelihood of D_obs and Y_obs
for clstr in pp.posterior_predictive['D_obs'].cluster:

    D_obs = trace.observed_data['D_obs'].sel({'cluster': clstr})
    Y_obs = trace.observed_data['Y_obs'].sel({'cluster': clstr})
    Y_obs_rel = 100 * Y_obs / Y_obs.sum(dim='serotype')
    D_sim = pp.posterior_predictive['D_obs'].sel({'cluster': clstr})
    p_true = 100 * pp.posterior_predictive['p_true'].sel({'cluster': clstr})

    fig,ax=plt.subplots(nrows=7, figsize=(8.3, 11.7/3*2))
    ax[0].plot(D_sim.simulation_date, D_obs, marker='o', markerfacecolor=None, markersize=2, color='black')
    ax[0].plot(D_sim.simulation_date, D_sim.mean(dim=['chain', 'draw']), color='red')
    ax[0].fill_between(D_sim.simulation_date, D_sim.quantile(dim=['chain', 'draw'], q=0.025),  D_sim.quantile(dim=['chain', 'draw'], q=0.975), color='red', alpha=0.3)
    ax[0].set_ylabel("DENV inc. (-)", fontsize=8)

    ax[1].plot(Y_obs.simulation_date, Y_obs.sum(dim='serotype'), marker='o', markerfacecolor=None, markersize=2, color='black')
    ax[1].set_ylabel("Serotyped (-)", fontsize=8)
    for i in range(1,5):
        ax[i+1].plot(Y_obs_rel.simulation_date, Y_obs_rel.sel({'serotype': i}), marker='o', markerfacecolor=None, markersize=2, color='black')
        ax[i+1].plot(p_true.simulation_date, p_true.sel({'serotype': i}).mean(dim=['chain', 'draw']), color='red')
        ax[i+1].fill_between(p_true.simulation_date, p_true.sel({'serotype': i}).quantile(dim=['chain', 'draw'], q=0.025), p_true.sel({'serotype': i}).quantile(dim=['chain', 'draw'], q=0.975), color='red', alpha=0.30)
        ax[i+1].set_ylabel(f"DENV {i} (%)", fontsize=8)
    ax[i+2].stackplot(p_true.simulation_date, jnp.transpose(p_true.mean(dim=['chain', 'draw']).values), colors=['black', 'red', 'green', 'blue'], labels=['DENV1', 'DENV2', 'DENV3', 'DENV4'])
    ax[i+2].legend(fontsize=6, loc=1, framealpha=1)
    ax[i+2].set_ylabel("Serotypes (%)", fontsize=8)
    plt.show()
    plt.close()


## seroprevalence
for clstr in pp.posterior_predictive['D_obs'].cluster:

    age_groups_labels = trace.observed_data['seroprev_obs'].age_group

    ### Find timesteps that are not nan
    valid_mask = jnp.any(~jnp.isnan(trace.observed_data['seroprev_obs'].sel({'cluster': clstr}).values), axis=1)
    indices = jnp.nonzero(valid_mask)[0]

    ### Initialise figure with correct ncols
    if len(indices) > 0:
        fig,axes = plt.subplots(ncols = len(indices), squeeze=False, sharey=True)
        ### Pull results and visualise
        axes = axes.ravel()
        for idx, ax in zip(indices, axes):
            data = 100*trace.observed_data['seroprev_obs'].sel({'cluster': clstr}).values[idx]
            model = 100*pp.posterior_predictive['seroprev_obs'].sel({'cluster': clstr}).mean(dim=['chain', 'draw']).values[idx]
            model_low = 100*pp.posterior_predictive['seroprev_obs'].sel({'cluster': clstr}).quantile(dim=['chain', 'draw'], q=0.025).values[idx]
            model_high = 100*pp.posterior_predictive['seroprev_obs'].sel({'cluster': clstr}).quantile(dim=['chain', 'draw'], q=0.975).values[idx]
            yerr = [model - model_low, model_high - model]
            ax.bar(age_groups_labels, data, color='black')
            ax.errorbar(age_groups_labels, model, yerr=yerr, fmt='o', color='red', capsize=4, elinewidth=1.5, linestyle='none')
            ax.plot(age_groups_labels, model, color='red')
            ax.set_ylim([0,100])
            ax.set_title(simulation_dates[idx].strftime('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=30)
            ax.set_xlabel('Age group')    
        axes[0].set_ylabel('Seroprevalence (%)')
        plt.tight_layout()
        plt.show()
        plt.close()


## latent states
for clstr in pp.posterior_predictive['D_obs'].cluster:

    data_dates = pp.posterior_predictive['beta_monthly'].data_date
    simulation_dates = trace.observed_data['D_obs'].simulation_date
    end_sim_data_index = calculate_simend_data_index(simulation_dates.values.tolist(), data_dates.values.tolist())

    fig,ax=plt.subplots(nrows=5, figsize=(8.3, 11.7/3*2))

    ### Transmission
    ax[0].plot(simulation_dates, pp.posterior_predictive['beta_monthly'].sel({'cluster': clstr}).mean(dim=['draw', 'chain'])[:end_sim_data_index+1], color='black')
    ax[0].fill_between(simulation_dates,
                       pp.posterior_predictive['beta_monthly'].sel({'cluster': clstr}).quantile(dim=['draw', 'chain'], q=0.025)[:end_sim_data_index+1],
                        pp.posterior_predictive['beta_monthly'].sel({'cluster': clstr}).quantile(dim=['draw', 'chain'], q=0.975)[:end_sim_data_index+1],
                        color='black', alpha=0.3)
    
    ### Homologous reinfection
    from cycler import cycler
    ax[1].set_prop_cycle(cycler('color', ['black', 'red', 'green', 'blue']))
    ax[1].plot(simulation_dates, 100*pp.posterior_predictive['p_hom_monthly'].sel({'cluster': clstr}).mean(dim=['draw', 'chain'])[:end_sim_data_index+1], label=['DENV1', 'DENV2', 'DENV3', 'DENV4'])
    for i in range(4):    
        ax[1].fill_between(simulation_dates,
                        pp.posterior_predictive['p_hom_monthly'].sel({'cluster': clstr}).quantile(dim=['draw', 'chain'], q=0.025)[:end_sim_data_index+1, i],
                            pp.posterior_predictive['p_hom_monthly'].sel({'cluster': clstr}).quantile(dim=['draw', 'chain'], q=0.975)[:end_sim_data_index+1, i],
                            color='red', alpha=0.3)
        ax[1].set_ylabel(r'$p_{hom,i}(t)$ (%)', fontsize=8)
        ax[1].legend(fontsize=8, loc=1, framealpha=1)

    ### Susceptibility slots per serotype
    ax[2].plot(simulation_dates, pp.posterior_predictive['S_expanded'].sel({'cluster': clstr, 'serotype': 1}).mean(dim=['chain', 'draw']).sum(dim=['age_group', 'n_prior_het_infections', 'infection_type']), color='black', label='DENV1')
    ax[2].plot(simulation_dates, pp.posterior_predictive['S_expanded'].sel({'cluster': clstr, 'serotype': 2}).mean(dim=['chain', 'draw']).sum(dim=['age_group', 'n_prior_het_infections', 'infection_type']), color='red', label='DENV2')
    ax[2].plot(simulation_dates, pp.posterior_predictive['S_expanded'].sel({'cluster': clstr, 'serotype': 3}).mean(dim=['chain', 'draw']).sum(dim=['age_group', 'n_prior_het_infections', 'infection_type']), color='green', label='DENV3')
    ax[2].plot(simulation_dates, pp.posterior_predictive['S_expanded'].sel({'cluster': clstr, 'serotype': 4}).mean(dim=['chain', 'draw']).sum(dim=['age_group', 'n_prior_het_infections', 'infection_type']), color='blue', label='DENV4')
    ax[2].plot(simulation_dates, pp.posterior_predictive['S_expanded'].sel({'cluster': clstr, 'serotype': 1, 'infection_type': 'hom'}).mean(dim=['chain', 'draw']).sum(dim=['age_group', 'n_prior_het_infections']), color='black', label='DENV1', linestyle='--')
    ax[2].plot(simulation_dates, pp.posterior_predictive['S_expanded'].sel({'cluster': clstr, 'serotype': 2, 'infection_type': 'hom'}).mean(dim=['chain', 'draw']).sum(dim=['age_group', 'n_prior_het_infections']), color='red', label='DENV2', linestyle='--')
    ax[2].set_ylabel('Susceptibility slots (-)', fontsize=8)

    ### Susceptibles by number of previous heterologous infections
    ax[3].stackplot(simulation_dates, jnp.transpose(pp.posterior_predictive['S_expanded'].sel({'cluster': clstr}).mean(dim=['chain', 'draw']).sum(dim=['age_group', 'serotype', 'infection_type']).values), colors=['black', 'red', 'orange', 'yellow', 'green'], labels=['0', '1', '2', '3', '4'])
    ax[3].legend(fontsize=8, loc=1, framealpha=1)
    ax[3].set_ylabel('Susceptibles (-)', fontsize=8)

    ### FOI
    ax[4].plot(simulation_dates, 100*pp.posterior_predictive['lambda_t'].sel({'cluster': clstr}).mean(dim=['draw', 'chain']), color='black')
    ax[4].fill_between(simulation_dates,
                       100*pp.posterior_predictive['lambda_t'].sel({'cluster': clstr}).quantile(dim=['draw', 'chain'], q=0.025),
                       100*pp.posterior_predictive['lambda_t'].sel({'cluster': clstr}).quantile(dim=['draw', 'chain'], q=0.975),
                       color='black', alpha=0.3)
    ax[4].set_ylabel('FOI (%)', fontsize=8)

    plt.show()
    plt.close()