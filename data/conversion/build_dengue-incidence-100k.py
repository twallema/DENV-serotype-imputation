
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

    # case data
    denv = pd.read_csv('../../data/interim/datasus_DENV-linelist/mun/DENV-serotypes_1996-2025_monthly_mun.csv', parse_dates=True)

    # Geography
    # >>>>>>>>>

    if region != 'CD_MUN':
            
        # get mapping
        muncipality_region_map = geography[['CD_MUN', f'{region}']]
        # dissolve to desired grouping
        geography = geography.dissolve(by=f'{region}', aggfunc={'POP': 'sum'}).reset_index()

        # Incidence
        # >>>>>>>>>

        # merge incidence with mapping
        denv = denv.merge(muncipality_region_map, on="CD_MUN", how="left")
        # define custom aggregation function to treat the Nans
        def nan_to_zero_sum(series):
            if series.isna().all():
                return float("nan")
            else:
                return series.fillna(0).sum()
        # group and aggregate
        denv = (
            denv.groupby([f"{region}", "date"])['DENV_total']
            .sum()
            .reset_index()
        )

    # Aggregate to per 100K
    # >>>>>>>>>>>>>>>>>>>>>

    # normalize total dengue cases to incidence per 100K
    denv = denv[[f'{region}', 'date', 'DENV_total']]
    denv = denv.merge(geography[[f'{region}', 'POP']], on=f'{region}', how="left")
    denv["DENV_per_100k"] = (
        denv["DENV_total"] / denv["POP"] * 1e5
    )

    # Save result
    # >>>>>>>>>>>

    denv[[f'{region}', 'date', 'DENV_per_100k']].to_csv(f'../interim/DENV_per_100k/DENV_per_100k_{name}.csv', index=False)
