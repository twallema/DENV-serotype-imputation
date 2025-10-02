
import pandas as pd
import geopandas as gpd

# Spatial aggregation levels
names = ['mun', 'rgi', 'rgint']
regions = ['CD_MUN', 'CD_RGI', 'CD_RGINT']


# Aggregate to the intermediate/immediate regions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

for region, name in zip(regions, names):

    # input check
    assert ((region != 'CD_MUN') | (region != 'CD_RGI') |  (region != 'CD_RGINT')), "'region' must be either 'CD_MUN', 'CD_RGI' or 'CD_RGINT''"

    # Reload raw data
    # >>>>>>>>>>>>>>>

    # geodata
    geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")

    # index-P data
    indexP = pd.read_csv('../../data/raw/indexP_monthlyclimate_allmuni.csv', parse_dates=True)[['code_muni', 'month', 'indexP']]
    indexP = indexP.rename(columns={'code_muni': 'CD_MUN'})  # harmonize municipality codes

    # Demographically weighted average
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if region != 'CD_MUN':
            
        # get municipality to region mapping
        municipality_region_map = geography[['CD_MUN', f'{region}']]

        # get muncipality population
        municipality_demograhpy = geography[['CD_MUN', 'POP']]

        # Merge index-P data with region map
        indexP = indexP.merge(municipality_region_map, on="CD_MUN", how="left")

        # Merge with population
        indexP = indexP.merge(municipality_demograhpy, on="CD_MUN", how="left")

        # Compute weighted indexP per region and month
        indexP = (
            indexP
            .groupby([f'{region}', "month"])
            .apply(lambda g: (g["indexP"] * g["POP"]).sum() / g["POP"].sum())
            .reset_index(name="indexP")
        )
    else:
        pass
    
    # Save result
    # >>>>>>>>>>>

    indexP[[f'{region}', 'month', 'indexP']].to_csv(f'../interim/indexP/indexP_{name}.csv', index=False)
