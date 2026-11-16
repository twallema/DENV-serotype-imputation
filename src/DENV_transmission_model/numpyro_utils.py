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
import pandas as pd
import polars as pl
import jax.numpy as jnp

abs_dir = os.path.dirname(__file__)

def get_incidence_data(data_dir, muni_clusters):
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

    Returns
    -------
    total_incidence : jnp.ndarray
        shape: (n_time, n_clusters)

    serotype_incidence : jnp.ndarray
        shape: (n_time, n_clusters, 4)
    """
    cluster_col = [c for c in muni_clusters.columns if c != "CD_MUN"][0]

    inc = pd.read_parquet(data_dir)
    inc = inc.merge(muni_clusters, on="CD_MUN", how="inner")
    inc = inc.groupby(["date", cluster_col], as_index=False)[
        ["DENV_total", "DENV_1", "DENV_2", "DENV_3", "DENV_4"]
    ].sum(min_count=1)

    total_incidence = jnp.array(inc.pivot(index="date", columns=cluster_col, values="DENV_total").to_numpy())
    serotype_incidence = jnp.array(inc.pivot(index="date", columns=cluster_col, values=["DENV_1", "DENV_2", "DENV_3", "DENV_4"]).to_numpy())

    return total_incidence, serotype_incidence

def get_seroprevalence_data(data_dir, age_groups, muni_clusters):

    serodata = pd.read_csv(data_dir)

    serodata = serodata[serodata['age_strata'] == True]

    # step 1: Identify what experiments happen in every cluster --> list containing a list for every cluster containing experiment names

    # Step 2: Identify what year the experiments happen in --> list containing a list for every cluster containing experiment names

    # loop over clusters and experiments

        ## get raw data and year

        ## get demography
    data_dir = os.path.join(abs_dir, '../../data/interim/demographics/population_mun-age_1998-2026.parquet')
    demography = _get_demography_by_yearofage(data_dir, muni_clusters, 2000)    # this function works and has 2000 as a placeholder year

        ## retrieve the age groups used in the experiment

    pass


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