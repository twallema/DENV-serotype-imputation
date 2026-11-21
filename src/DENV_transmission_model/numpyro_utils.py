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

import os
import sys
import time
import pandas as pd
import polars as pl

import jax
import jax.numpy as jnp
import optax
from numpyro.infer.util import initialize_model
from numpyro.infer import init_to_median, init_to_value

abs_dir = os.path.dirname(__file__)


##################
## Getting data ##
##################

def get_incidence_data(data_dir, muni_clusters, data_start):
    """
    Load municipality-level dengue incidence and aggregate it into clusters.

    Parameters
    ----------
    data_dir : str or path
        Path to incidence data per municipality and month

    muni_clusters : pandas.DataFrame
        Mapping of municipalities to clusters. Must contain the column
        ``CD_MUN`` plus exactly one additional column holding the cluster
        identifier.

    data_start: datetime.datetime
        When the data should start.
        Equate to the transmission model simulation start

    Returns
    -------

    data_dates: list containing np.datetime
        shape: (n_time)
        dates from data_start to the end of the available data

    total_incidence : jnp.ndarray
        shape: (n_time, n_clusters)

    serotype_incidence : jnp.ndarray
        shape: (n_time, n_clusters, 4)
    """

    cluster_col = [c for c in muni_clusters.columns if c != "CD_MUN"][0]

    inc = pd.read_parquet(data_dir)
    inc = inc[inc['date'] >= data_start]
    inc = inc.merge(muni_clusters, on="CD_MUN", how="inner")
    inc = inc.groupby(["date", cluster_col], as_index=False)[
        ["DENV_total", "DENV_1", "DENV_2", "DENV_3", "DENV_4"]
    ].sum(min_count=1)

    data_dates = pd.DatetimeIndex(inc['date']).to_pydatetime().tolist()

    total_incidence = jnp.array(inc.pivot(index="date", columns=cluster_col, values="DENV_total").to_numpy())
    serotype_incidence = jnp.array(inc.pivot(index="date", columns=cluster_col, values=["DENV_1", "DENV_2", "DENV_3", "DENV_4"]).to_numpy())

    return data_dates, total_incidence, serotype_incidence


def get_seroprevalence_data(data_dir, age_groups, muni_clusters):
    """
    Load serosurvey results and convert them to the desired age groups.

    Every serosurvey ("experiment") conducted in one or more municipalities of
    a cluster is converted to the desired age groups by demographic weighting:
    the reported age strata of the survey are combined into every desired age
    group, weighting each stratum by the population in the overlap between the
    study's age bracket and the desired age bracket. Population by year of age
    is taken from the demography data of the year the survey was performed.
    Seroprevalence is returned as a percentage.

    Parameters
    ----------
    data_dir : str or path
        Path to the seroprevalence data

    age_groups : list of tuple
        Desired age groups, as (low, high) pairs, e.g. [(0, 1), (1, 5)].

    muni_clusters : pandas.DataFrame
        Mapping of municipalities to clusters. Must contain the column
        ``CD_MUN`` plus exactly one additional column holding the cluster
        identifier.

    Returns
    -------
    seroprevalence : dict
        Maps the cluster identifier to a pandas.DataFrame indexed on
        ``age_groups`` (labelled ``[{start}, {end})``), whose columns are the
        experiments conducted in that cluster (named ``{study_id}_{year}``)
        and whose values are the seroprevalence percentages.
    """
    
    max_age = 80 # highest single year of age available in the demography data

    serodata = pd.read_csv(data_dir)

    serodata = serodata[serodata['age_strata'] == True]

    cluster_col = [c for c in muni_clusters.columns if c != "CD_MUN"][0]

    # map every survey onto its cluster
    serodata = serodata.merge(muni_clusters, on="CD_MUN", how="inner")

    # parse the age brackets of every reported age stratum --> (age_low, age_high);
    # missing bounds are interpreted as open-ended brackets (0-... or ...-80+)
    def parse_bracket(row):
        low, med, high = row["age_low"], row["age_med"], row["age_high"]
        if pd.notna(low) and pd.notna(high):
            return (low, high)
        if pd.notna(low):
            return (low, max_age)
        if pd.notna(high):
            return (0, high)
        if pd.notna(med):
            return (med, med)
        return None

    serodata["bracket"] = serodata.apply(parse_bracket, axis=1)
    serodata = serodata[serodata["bracket"].notna() & serodata["seroprev_med"].notna()]

    # the demography matrices share one column order: ascending cluster identifiers
    clusters = sorted(muni_clusters[cluster_col].unique())

    # labels for the desired age groups
    labels = [f"[{start}, {end})" for start, end in age_groups]

    # loop over clusters and experiments
    seroprevalence = {}
    for cluster, cluster_data in serodata.groupby(cluster_col):

        experiments = {}
        for (study_id, start_year), strata in cluster_data.groupby(["study_id", "start_year"]):

            # demography (population by year of age) of the cluster in the survey year
            demographics_path = os.path.join(abs_dir, '../../data/interim/demographics/population_mun-age_1998-2026.parquet')
            demography = _get_demography_by_yearofage(demographics_path, muni_clusters, int(start_year))[:, clusters.index(cluster)]

            # demographically-weighted seroprevalence in every desired age group
            weighted_seroprev = {}
            for label, age_group in zip(labels, age_groups):
                weights, values = [], []
                for _, stratum in strata.iterrows():
                    stratum_low, stratum_high = stratum["bracket"]
                    a_low = max(int(stratum_low), age_group[0])
                    a_high = min(int(stratum_high) + 1, age_group[1], max_age + 1)
                    if a_high <= a_low:
                        continue
                    weights.append(float(demography[a_low:a_high].sum()))
                    values.append(stratum["seroprev_med"])
                if sum(weights) > 0:
                    weighted_seroprev[label] = sum(w * p for w, p in zip(weights, values)) / sum(weights)
                else:
                    weighted_seroprev[label] = float('nan') # age group not covered by the experiment

            experiments[f"{int(start_year)}_{study_id}"] = pd.Series(weighted_seroprev)

        # merge the experiments in this cluster into one dataframe indexed on the age_groups
        seroprevalence[cluster] = pd.DataFrame(experiments)

    return seroprevalence


def _get_demography_by_yearofage(data_dir, muni_clusters, year):
    """
    Load population counts by year of age for all clusters in a given year.

    Parameters
    ----------
    data_dir : str or path
        Path to population data per municipality and year of age

    muni_clusters : pandas.DataFrame
        Mapping of municipalities to clusters. Must contain the column
        ``CD_MUN`` plus exactly one additional column holding the cluster
        identifier.

    year : int
        Calendar year for which to extract the population pyramid.

    Returns
    -------
    demography : jnp.ndarray
        shape: (81, n_clusters). Population by year of age from 0 to 80,
        summed over the municipalities belonging to each cluster.
    """

    cluster_col = [c for c in muni_clusters.columns if c != "CD_MUN"][0]

    demo = (
        pl.scan_parquet(data_dir)
        .filter(pl.col("year") == year)
        .sort([ "CD_MUN", "age"])
        .collect()
        .to_pandas()
    )

    demo = demo.merge(muni_clusters, on="CD_MUN", how="inner")

    demo = demo.groupby(by=[cluster_col, 'age'])['population'].sum().reset_index()

    return jnp.array(demo.pivot(index="age", columns=cluster_col, values="population").to_numpy()) # (= jnp array of shape (81, n_clusters), containing population by year of age from 0 to 80 for all clusters)


#####################
## Finding the MAP ##
#####################

def find_map(model, model_kwargs, n_preoptim=1000, lr=1e-2, map_params=None):
    """
    Find the MAP estimate of a NumPyro model using Adam optimization.

    Parameters
    ----------
    model : callable
        NumPyro model.

    model_kwargs : dict
        Keyword arguments passed to `training_model`.

    n_preoptim : int
        Number of Adam optimization steps.

    lr : float
        Adam learning rate.

    Returns
    -------
    map_params : dict
        MAP estimate in the constrained parameter space used by the NumPyro model.
    """

    # --------------------------------------------------
    # Initialize model in unconstrained space
    # --------------------------------------------------

    rng_key = jax.random.PRNGKey(int(time.time()))

    if map_params:
        init_strategy=init_to_value(values=map_params)
    else:
        init_strategy=init_to_median()

    init_params_info, potential_fn, postprocess_fn, _ = (
        initialize_model(
            rng_key,
            model,
            model_kwargs=model_kwargs,
            init_strategy=init_strategy
        )
    )

    z = init_params_info.z

    # --------------------------------------------------
    # Initialise Adams optimizer
    # --------------------------------------------------

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(z)

    # --------------------------------------------------
    # Define ptimization step
    # --------------------------------------------------

    @jax.jit
    def step(z, opt_state):

        potential_energy, grads = jax.value_and_grad(potential_fn)(z)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            z,
        )

        z = optax.apply_updates(
            z,
            updates,
        )

        return z, opt_state, potential_energy

    # --------------------------------------------------
    # Optimize and print progress
    # --------------------------------------------------

    for i in range(n_preoptim+1):

        z, opt_state, pe = step(z, opt_state)

        if i % 100 == 0:
            print(i, float(pe))
            sys.stdout.flush()

    # --------------------------------------------------
    # Transform back to constrained space
    # --------------------------------------------------

    map_params = postprocess_fn(z)

    return map_params

# TODO: sequential optimiser