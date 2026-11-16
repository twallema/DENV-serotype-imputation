import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
from datetime import datetime
import matplotlib.pyplot as plt
from DENV_transmission_model.jax_forward_sim_model import DengueModel
from DENV_transmission_model.numpyro_models import single_cluster_training
from DENV_transmission_model.numpyro_utils import get_incidence_data, get_seroprevalence_data

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
simulation_start = datetime(1998, 9, 1)
## when to introduce DENV 3 and DENV 4
intro_dates_DENV3 = [datetime(2000, 1, 1),]
intro_dates_DENV4 =[datetime(2007, 1, 1),]


# initialise the model
model = DengueModel(age_groups, muni_clusters, simulation_start, intro_dates_DENV3, intro_dates_DENV4)
n_months = len(model.months)


# load the fitting data
## total incidence
data_dir = os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/DENV-1998_2026-month-mun-no_diagnostics.parquet')
total_incidence, serotype_incidence = get_incidence_data(data_dir, muni_clusters)
## seroprevalence
data_dir = os.path.join(abs_dir, '../../data/interim/seroprevalence/seroprevalence.csv')
seroprevalence = get_seroprevalence_data(data_dir, age_groups, muni_clusters)


# load the numpyro model

# pre-optimize the MAP

