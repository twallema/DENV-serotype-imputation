
"""
This scripts converts the municipality-level human development index to the regions by computing a population-weighted average
"""

import os
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# make output folder
abs_dir = os.path.dirname(__file__)
output_folder = os.path.join(abs_dir, f'../interim/human_development_index')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Spatial aggregation levels
names = ['rgi', 'rgint']
regions = ['CD_RGI', 'CD_RGINT']

for name,region in zip(names,regions):

    # load data
    hdi = pd.read_csv(os.path.join(abs_dir, '../raw/demographics/human_development_index.csv'))

    # convert data to numerical
    numeric_cols = ["IDHM 2010", "IDHM Renda 2010", "IDHM Longevidade 2010", "IDHM Educação 2010"]
    hdi[numeric_cols] = hdi[numeric_cols].replace(",", ".", regex=True).astype(float)

    # split state and municipality
    hdi["ABBREV_UF"] = hdi["Município"].str.extract(r"\(([A-Z]{2})\)$")
    hdi["NM_MUN"] = hdi["Município"].str.replace(r"\s*\([A-Z]{2}\)$", "", regex=True)

    # build lookup dictionary
    geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
    muncipality_region_map = geography[['CD_MUN', f'{region}']].set_index('CD_MUN').to_dict()[f'{region}']

    state_abbreviations = {"Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM", "Bahia": "BA", "Ceará": "CE", "Distrito Federal": "DF", "Espírito Santo": "ES", "Goiás": "GO", "Maranhão": "MA",
        "Mato Grosso": "MT", "Mato Grosso do Sul": "MS", "Minas Gerais": "MG", "Pará": "PA", "Paraíba": "PB", "Paraná": "PR", "Pernambuco": "PE", "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
        "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR", "Santa Catarina": "SC", "São Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO",
    }

    lookup = geography[['CD_MUN', 'NM_MUN', 'NM_UF']]
    lookup["ABBREV_UF"] = lookup["NM_UF"].map(state_abbreviations)

    # lowercode unicode only
    lookup["NM_MUN"] = lookup["NM_MUN"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()
    hdi["NM_MUN"] = hdi["NM_MUN"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()

    # attach matching municipality codes
    hdi = pd.merge(
        hdi, 
        lookup[['CD_MUN', 'NM_MUN', 'ABBREV_UF']], 
        on=['NM_MUN', 'ABBREV_UF'], 
        how='left'
    )[['CD_MUN', 'IDHM 2010']]

    # attach region code
    hdi[f'{region}'] = hdi['CD_MUN'].map(muncipality_region_map)

    # attach population count in 2010
    pop = pl.scan_parquet("../../data/interim/demographics/population_mun-age_1999-2026.parquet").filter(pl.col("year") == 2010).group_by("CD_MUN").agg(pl.col("population").sum()).collect().to_pandas()
    hdi = pd.merge(
        hdi, 
        pop, 
        on=['CD_MUN'], 
        how='left'
    )

    # compute the population-weighted average grouped by region
    hdi = hdi.groupby(f"{region}").apply(
        lambda x: (x["IDHM 2010"] * x["population"]).sum() / x["population"].sum(),
        include_groups=False
    ).reset_index(name="weighted_IDHM_2010")

    # change the name
    hdi = hdi.rename(columns={'weighted_IDHM_2010': 'hdi_2010'})

    # visualise result
    ## dissolve geography and compute area
    geography = geography.to_crs("EPSG:5880")
    geography = geography.dissolve(by=f'{region}').reset_index()
    geography = geography.merge(hdi[[f'{region}', 'hdi_2010']], how='left')

    # save result
    geography[[f'{region}', 'hdi_2010']].to_csv(os.path.join(output_folder, f'hdi_2010_{name}.csv'), index=False)

    # visualise on a map
    fig, ax = plt.subplots()
    geography.plot(
        column="hdi_2010",
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        ax=ax,
    )
    ax.axis("off")
    ax.set_title("Human development index (2010)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'hdi_2010_{name}.svg'))
    plt.close()