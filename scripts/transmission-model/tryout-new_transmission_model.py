import numpy as np
import jax.numpy as jnp
from datetime import datetime
from DENV_transmission_model.jax_forward_sim_model import DengueModel
import matplotlib.pyplot as plt

# settings
municipalities = [2611606, 2927408] # Recife and Salvador
age_groups = [(0, 1), (1, 6), (6, 12), (12, 18), (18, 25), (25,35), (35,55), (55,120)]
n_munis = len(municipalities)
n_age_groups = len(age_groups)
n_serotypes = 4

simulation_start = datetime(1998, 9, 1)

intro_dates_DENV3 = [datetime(2007, 1, 1), datetime(2009, 1, 1)]
intro_dates_DENV4 =[datetime(2010, 1, 1), datetime(2010, 1, 1)]

# initialise the model
model = DengueModel(age_groups, municipalities, simulation_start, intro_dates_DENV3, intro_dates_DENV4)

n_months = len(model.months)

# build initial conditions
## parameters
f_P = 0.25 * jnp.ones([n_munis, n_age_groups])
f_P2 = 0.33 * jnp.ones([n_munis, n_age_groups])
pi_mono2 = 0.33 * jnp.ones([n_munis, n_age_groups])
I_estim = jnp.array([1e-5, 1e-5])
p0 = jnp.array([[0.75, 0.25, 0, 0], [0.75, 0.25, 0, 0]])
## build
pi_d = model.infection_history_from_foi(jnp.array([0.05, 0.05]))
S0 = model.build_initial_susceptibles(f_P, pi_d, pi_mono2) # (n_munis, n_age_groups, 16)
P0 = model.build_initial_crossprotection(f_P, pi_d, f_P2)
I0 = model.build_initial_infected(I_estim, p0, pi_d)

# define any remaining parameters
## simulation parameters
stepsize = 0.25
## core epi parameters
gamma = 0.5
omega = 18
## Probability of observing an infection
kappa = jnp.ones([n_munis, n_age_groups, 4, n_serotypes]) # (cluster, age_group, infection_degree, serotype)
## time-varying transmission
mu_beta = jnp.array([jnp.log(2.0), jnp.log(2.0)])
A_beta = jnp.array([1, 1])
phi_beta = jnp.array([jnp.pi/2, jnp.pi/2])
season = A_beta[None, :] * jnp.cos(2 * jnp.pi * jnp.arange(len(model.months))[:, None] / 12 - phi_beta[None, :])    # --> maybe to helper function
beta_monthly = jnp.exp(mu_beta[None, :] + season)
## homologous reinfection
p_hom = jnp.array([0.8, 0.8, 0.8, 0.8])
p_hom_monthly = jnp.broadcast_to(p_hom, (n_months, n_munis, n_serotypes))

# simulate model
simout = model.sim(kappa, gamma, omega, beta_monthly[:12,...], p_hom_monthly[:12,...], S0, I0, P0, datetime(1999,9,1), stepsize)

# wrap output in an xarray
simout = model.simout_to_xarray(simout)

# visualise output
fig,ax = plt.subplots(nrows=4, ncols=1, sharex=True)
ax[0].stackplot(simout.date, np.transpose(simout["S_expanded"].sum(dim=["age_group", "serotype", "infection_type"]).sel(CD_MUN=2611606).values))
ax[1].plot(simout.date, simout["lambda_t"].sel(CD_MUN=2611606).sum(dim='serotype'))
ax[2].plot(simout.date, simout["dI_obs"].sel(CD_MUN=2611606))
ax[3].stackplot(simout.date, np.transpose(simout["p_true"].sel(CD_MUN=2611606).values), labels=[1,2,3,4])
ax[3].legend()
plt.show()
plt.close()