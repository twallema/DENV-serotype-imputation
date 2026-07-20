
import pandas as pd
import polars as pl
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
    denv = pl.scan_parquet("../../data/interim/datasus_DENV-linelist/DENV-1999_2026-month-mun-no_diagnostics.parquet").collect().to_pandas()

    # population data
    pop = pl.scan_parquet("../../data/interim/demographics/population_mun-age_1999-2025.parquet").group_by(["CD_MUN", "year"]).agg(pl.col("population").sum()).sort(by=["CD_MUN", "year"]).collect().to_pandas()

    # Geography
    # >>>>>>>>>

    if region != 'CD_MUN':
            
        # get mapping
        muncipality_region_map = geography[['CD_MUN', f'{region}']]

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

        # convert population-by-year to desired mapping
        pop = pop.merge(muncipality_region_map, on="CD_MUN", how="left")
        pop = pop.groupby([f"{region}", "year"])['population'].sum().reset_index()


        # Aggregate to per 100K
        # >>>>>>>>>>>>>>>>>>>>>

        # attach the population column to dengue dataset
        denv['year'] = denv['date'].dt.year
        denv = pd.merge(denv, pop, on=[f"{region}", 'year'], how='left')

        # compute cases per 100K
        denv["DENV_per_100k"] = denv["DENV_total"] / denv["population"] * 1e5

        # Save result
        # >>>>>>>>>>>

        denv[[f'{region}', 'date', 'DENV_per_100k']].to_csv(f'../interim/DENV_per_100k/DENV_per_100k_{name}.csv', index=False)
