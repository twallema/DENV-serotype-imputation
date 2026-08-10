
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Spatial aggregation levels
names = ['rgi', 'rgint']
regions = ['CD_RGI', 'CD_RGINT']

for name,region in zip(names,regions):

    # geodata
    geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")
    muncipality_region_map = geography[['CD_MUN', f'{region}']].set_index('CD_MUN').to_dict()[f'{region}']

    # land cover data
    land_cover = pd.read_excel("../../data/raw/land_cover/Municipality_cover_and_land_use_with_GEOCODE_24fev2026.xlsx")

    # attach region index
    land_cover[f'{region}'] = land_cover['code_muni'].map(muncipality_region_map)

    # sum "4.2 Urban Area" per region
    land_cover = land_cover[land_cover['class_level_2'] == '4.2. Urban Area'].groupby(by=f'{region}')[[str(year) for year in range(1998, 2025)]].sum()

    # average over years 1998-2026
    land_cover = land_cover.reset_index().melt(id_vars=f'{region}',  var_name='year', value_name='urban_area').groupby(by=f"{region}")['urban_area'].mean().reset_index() 
    land_cover['urban_area'] *= 10000

    # dissolve geography and compute area
    geography = geography.to_crs("EPSG:5880")
    geography = geography.dissolve(by=f'{region}').reset_index()
    geography["area"] = geography.geometry.area
    
    # merge area to the land cover data
    geography = geography.merge(land_cover[[f'{region}', 'urban_area']], how='left')

    # normalise urban area
    geography['fraction_urban'] = geography['urban_area'] / geography['area']

    # save result
    geography[[f'{region}', 'fraction_urban']].to_csv(f'../interim/land_cover/fraction_urban_land_{name}.csv', index=False)

    # visualise on a map
    fig, ax = plt.subplots()
    geography['plot_fraction_urban'] = np.log10(geography['fraction_urban'])
    geography.plot(
        column="plot_fraction_urban",          # color regions by cluster label
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        ax=ax,
    )
    ax.axis("off")
    ax.set_title("Fraction urban land (log10)")
    plt.tight_layout()
    plt.savefig(f'../interim/land_cover/fraction_urban_land_{name}.svg')
    plt.close()