
"""
This scripts converts the municipality-level density-weighted mean density to the regions by computing a population-weighted average
"""

import os
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# make output folder
abs_dir = os.path.dirname(__file__)
output_folder = os.path.join(abs_dir, f'../interim/demographics')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Spatial aggregation levels
names = ['rgi', 'rgint']
regions = ['CD_RGI', 'CD_RGINT']

for name,region in zip(names,regions):

    # load data
    dens = pd.read_csv(os.path.join(abs_dir, '../raw/demographics/pop_totals_density.csv'))[['code_muni', 'total_population', 'density_weighted_mean_density']]
    dens = dens.rename(columns={'code_muni': 'CD_MUN'})

    # build lookup dictionary
    geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
    muncipality_region_map = geography[['CD_MUN', f'{region}']].set_index('CD_MUN').to_dict()[f'{region}']

    # attach regions
    dens[f"{region}"] = dens["CD_MUN"].map(muncipality_region_map)

    # compute population-weighted average in the region
    dens = dens.groupby(f"{region}").apply(
        lambda x: (x["density_weighted_mean_density"] * x["total_population"]).sum() / x["total_population"].sum(),
        include_groups=False
    ).reset_index(name="density_weighted_mean_density")


    # visualise result
    ## dissolve geography and compute area
    geography = geography.to_crs("EPSG:5880")
    geography = geography.dissolve(by=f'{region}').reset_index()
    geography = geography.merge(dens[[f'{region}', 'density_weighted_mean_density']], how='left')

    # save result
    geography[[f'{region}', 'density_weighted_mean_density']].to_csv(os.path.join(output_folder, f'density_weighted_mean_density_{name}.csv'), index=False)

    # visualise on a map
    fig, ax = plt.subplots()
    geography['plot_density_weighted_mean_density'] = np.log10(geography['density_weighted_mean_density'])
    geography.plot(
        column="plot_density_weighted_mean_density",
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        ax=ax,
    )
    ax.axis("off")
    ax.set_title("Density-weighted mean density (log10)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'density_weighted_mean_density_{name}.svg'))
    plt.close()