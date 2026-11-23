"""
This script contains all functions related to the Bayesian numpyro model

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

import jax
import numpyro
import jax.numpy as jnp
import jax.scipy as jsp
from patsy import dmatrix
import numpyro.distributions as dist
from functools import partial

import matplotlib.pyplot as plt

####################
## Training model ##
####################

single_cluster_training_model_RV_dims = {
    'pi_d': ['cluster', 'age_group', 'prev_heterological_infections_pi_d'],
    'f1': ['data_date', 'cluster'],
    'f2': ['data_date', 'cluster'],
    'f3': ['data_date', 'cluster'],
    'f4': ['data_date', 'cluster'],
    'p_hom_monthly': ['data_date', 'cluster', 'serotype'],
    'eps_raw': ['data_date', 'cluster'],
    'beta_monthly': ['data_date', 'cluster'],
    'beta_monthly_season': ['data_date', 'cluster'],
    'beta_monthly_shock': ['data_date', 'cluster'],
    'log_or_serotype': ['serotype'],
    'kappa': ['cluster', 'age_group', 'current_het_infection', 'serotype', 'infection_type'],
    'lambda_t': ['simulation_date', 'cluster'],
    'dI_obs': ['simulation_date', 'cluster', 'age_group'],
    'p_true': ['simulation_date', 'cluster', 'serotype'],
    'p_obs': ['simulation_date', 'cluster', 'serotype'],
    'seroprev_model': ['simulation_date', 'cluster', 'age_group'],
    'S_expanded': ['simulation_date', 'cluster', 'age_group', 'n_prior_het_infections', 'serotype', 'infection_type'],
    'D_obs': ['simulation_date', 'cluster'],
    'Y_obs': ['simulation_date', 'cluster', 'serotype'],
    'seroprev_obs' : ['simulation_date', 'cluster', 'age_group']
}


def single_cluster_training_model(transmission_model, simulation_dates, total_incidence, serotype_incidence, seroprevalence, data_dates):

    n_serotypes = 4
    N, M = 1, len(transmission_model.age_groups)
    S, D = len(simulation_dates), len(data_dates)

    end_sim_data_index = calculate_simend_data_index(simulation_dates, data_dates)

    # build initial conditions
    ## parameters
    pi_mono2_raw = numpyro.sample("pi_mono2_raw", dist.Beta(2, 2).expand([N,]))
    pi_mono2 = numpyro.deterministic("pi_mono2", jnp.broadcast_to(pi_mono2_raw[:, None], (N, M)))         

    f_P_raw = numpyro.sample("f_P_raw", dist.TransformedDistribution(dist.Normal(loc=jsp.special.logit(0.50), scale=1).expand([N,]), dist.transforms.SigmoidTransform()))
    f_P = numpyro.deterministic("f_P", jnp.broadcast_to(f_P_raw[:, None], (N, M)))

    f_P2_raw = numpyro.sample("f_P2_raw", dist.TransformedDistribution(dist.Normal(loc=jsp.special.logit(0.50), scale=1).expand([N,]), dist.transforms.SigmoidTransform()))
    f_P2 = numpyro.deterministic("f_P2", jnp.broadcast_to(f_P2_raw[:, None], (N, M)))

    I_estim = numpyro.sample("I_estim", dist.HalfNormal(1e-4).expand([N,]))

    foi_year = numpyro.sample("foi_year", dist.TransformedDistribution(dist.Normal(loc=jsp.special.logit(0.05), scale=1/3).expand([N,]), dist.transforms.SigmoidTransform()))   # 5% +- 2%

    p0 = jnp.mean(transmission_model.seeding[:6,:,:], axis=0)   # first six months used to estimate initial serotype distribution

    ## build
    pi_d = numpyro.deterministic("pi_d", transmission_model.infection_history_from_foi(foi_year))
    S0 = transmission_model.build_initial_susceptibles(f_P, pi_d, pi_mono2) # (n_munis, n_age_groups, 16)
    P0 = transmission_model.build_initial_crossprotection(f_P, pi_d, f_P2)
    I0 = transmission_model.build_initial_infected(I_estim, p0, pi_d)

    # build time-dependent parameters
    ## homologeous reinfection probability
    spline_basis = jnp.asarray(dmatrix(f"bs(x, df={int(D/3)-1}, degree=3, include_intercept=False)", {"x": range(D)}, return_type="dataframe").to_numpy())

    baseline = 0
    f1_coeff = numpyro.sample("f1_coeff", dist.Normal(baseline, 1).expand([spline_basis.shape[1], N]))     # TODO: RW(2) smoothness prior + force intercept at 50%?
    f1 = numpyro.deterministic("f1", jax.nn.sigmoid(spline_basis @ f1_coeff))
    
    f2_coeff = numpyro.sample("f2_coeff", dist.Normal(baseline, 1).expand([spline_basis.shape[1], N]))
    f2 = numpyro.deterministic("f2", jax.nn.sigmoid(spline_basis @ f2_coeff))
          
    f3 = jnp.zeros((D,N), dtype=f1.dtype)

    f4 = jnp.zeros((D,N), dtype=f1.dtype)

    p_hom_monthly = numpyro.deterministic("p_hom_monthly", jnp.stack([f1, f2, f3, f4], axis=-1))

    ## seasonality
    ### Global pattern
    mu_beta = numpyro.sample("mu_beta", dist.Normal(jnp.log(2.5), 1).expand([N,]))
    A_beta = numpyro.sample("A_beta", dist.HalfNormal(1).expand([N,]))
    phi_beta = numpyro.sample("phi_beta", dist.Normal((2 * jnp.pi * 3)/ 12, 1).expand([N,]))  # peaks in month 3 (march)
    data_months = jnp.arange(D, dtype=jnp.float32)
    season = A_beta[:, None] * jnp.cos(2.0 * jnp.pi * data_months[None, :] / 12 - phi_beta[:, None]) # Shape: (n_clusters, n_months)
    ### AR(1) deviations
    rho_ar_beta = 0.5
    sigma_ar_beta = numpyro.sample("sigma_ar_beta", dist.HalfNormal(1/100))
    eps_raw = numpyro.sample("eps_raw", dist.Normal(0, 1).expand([D, N]))
    u0 = jnp.zeros((N,), dtype=jnp.float32)
    _, u_seq = jax.lax.scan(partial(ar1_step, rho_ar_beta, sigma_ar_beta), u0, eps_raw[1:])
    u = jnp.concatenate([u0[None, :], u_seq], axis=0) # u has shape (n_months, n_clusters)
    ### Combine
    log_beta_monthly = mu_beta[None, :] + season.T + u
    beta_monthly = numpyro.deterministic("beta_monthly", jnp.exp(log_beta_monthly))
    ### Save parts for diagnosis
    numpyro.deterministic("beta_monthly_season", jnp.exp(mu_beta[None, :] + season.T))
    numpyro.deterministic("beta_monthly_shock", jnp.exp(u))

    # Reported fraction (cluster x age_group x degree x serotype x homologous indicator)
    kappa0_logit = numpyro.sample("kappa0_logit", dist.Normal(jsp.special.logit(1/10), 1))                          # intercept
    is_34 = jnp.array([0, 0, 1, 1], dtype=jnp.float32)                                                              # indicator prim/sec versus tert/quart
    log_or_34 = numpyro.sample("log_or_34", dist.Normal(-3, 1/3))                                                   # log-OR detecting prim/sec versus tert/quart
    log_or_serotype_raw = numpyro.sample("log_or_serotype_raw", dist.Normal(0, 1/10).expand([n_serotypes - 1]))     # log-OR detecting serotypes (vs. baseline serotype)
    log_or_serotype = numpyro.deterministic("log_or_serotype", jnp.concatenate([jnp.zeros((1,), dtype=log_or_serotype_raw.dtype), log_or_serotype_raw]))
    is_hom = jnp.array([1, 0], dtype=jnp.float32)                                                           # indicator for homologous infection
    log_or_hom = numpyro.sample("log_or_hom", dist.Normal(-3, 1/3))                                        # TODO: check if applied right (to)
    # Shape: (degree x serotype x homologous indicator)
    logit_kappa = kappa0_logit + log_or_34 * is_34[:, None, None] + log_or_serotype[None, :, None] + log_or_hom * is_hom[None, None, :]
    # Broadcast logit_kappa over the clusters and age groups
    kappa = numpyro.deterministic("kappa", jax.nn.sigmoid(jnp.broadcast_to(logit_kappa[None, None, :, :, :], (N, M, 4, n_serotypes, 2))))

    # build clinical parameters
    gamma = 0.5
    omega = numpyro.sample("omega", dist.LogNormal(3.2, 0.33))     

    # run transmission model
    simout = transmission_model.sim(kappa, gamma, omega, beta_monthly[:end_sim_data_index+1], p_hom_monthly[:end_sim_data_index+1], S0, I0, P0, simulation_dates[-1])
    S_t, I_t, P_t, _, _, _, S_expanded, dI_obs, p_obs, p_true, lambda_t = simout

    # save simulation output
    lambda_t = numpyro.deterministic("lambda_t", jnp.sum(lambda_t, axis=-1)) 
    dI_obs = numpyro.deterministic("dI_obs", dI_obs)
    p_obs = numpyro.deterministic("p_obs", p_obs)
    p_true = numpyro.deterministic("p_true", p_true)
    S_expanded = numpyro.deterministic("S_expanded", S_expanded)
    seroprev_model = numpyro.deterministic("seroprev_model", 100 - 100 * S_t[..., 0] / (jnp.sum(S_t, axis=-1) + jnp.sum(I_t, axis=-1) + jnp.sum(P_t, axis=-1)))

    # build observations
    ## observed cases (Poisson)
    #with numpyro.handlers.scale(scale=1/total_incidence[:end_sim_data_index+1].size):
    numpyro.sample("D_obs", dist.Poisson(rate=jnp.sum(dI_obs, axis=-1)), obs=total_incidence[:end_sim_data_index+1])

    ## serotyped cases
    #### overdispersion
    inv_phi = numpyro.sample("inv_phi", dist.HalfNormal(1/100))
    phi = numpyro.deterministic("phi", 1/inv_phi)
    ### concentration
    alpha = phi * p_obs
    ### Observed Dirichlet-Multinomial likelihood
    total_count = jnp.sum(serotype_incidence[:end_sim_data_index+1], axis=-1)
    with numpyro.handlers.scale(scale=1):
        numpyro.sample("Y_obs", dist.DirichletMultinomial(total_count=total_count, concentration=alpha), obs=serotype_incidence[:end_sim_data_index+1])

    ## seroprevalence
    obs = (seroprevalence / 100.0)[:end_sim_data_index+1]
    mask = ~jnp.isnan(obs)  # no nans allowed in likelihood
    obs = jnp.where(mask, obs, 0.5)
    with numpyro.handlers.scale(scale=10):
        numpyro.sample("seroprev_obs", dist.TransformedDistribution(dist.Normal(loc=jsp.special.logit(seroprev_model/100)[:end_sim_data_index+1], scale=0.10), dist.transforms.SigmoidTransform()).mask(mask), obs=obs)

    pass


def ar1_step(rho, sigma, u_prev, eps_t):
  u_curr = rho * u_prev + sigma * eps_t
  u_curr = jnp.asarray(u_curr, dtype=u_prev.dtype)
  return u_curr, u_curr


def calculate_simend_data_index(simulation_dates, data_dates):
    """
    A function calculating the last index in the dataset for which there is a matching disease transmission model output

    The start of the simulation and start of data must coincide
    The end of the simulation must fall before the end of the data

    Parameters
    ----------

    simulation_dates: list containing datetime
        Dates for which the disease transmission model is simulated

    data_dates: list containing datetime
        Dates for which there is data to fit the disease transmission model to

    Returns
    -------

    simend_data_index: int
        Last date index with simulation output
    """
    assert simulation_dates[0] == data_dates[0], 'start of simulation and data must coincide'
    assert simulation_dates[-1] <= data_dates[-1], 'end of simulation must fall before end of data'

    return data_dates.index(simulation_dates[-1])
