
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Spatial aggregation levels
names = ['mun', 'rgi', 'rgint']
regions = ['CD_MUN', 'CD_RGI', 'CD_RGINT']


# Load the human footprint data and average out years
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# geodata
geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")

# index-P data
hfp = pd.read_csv('../../data/raw/skinner_etal_2023/full_dataset.csv', parse_dates=True)[['CD_MUN', 'year', 'human_footprint']]

# average out years
hfp = pd.DataFrame(hfp.groupby(by=['CD_MUN'])['human_footprint'].mean()).reset_index()

# Replace the truncated area code with the official area code
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Step 1 - Get the official municipality codes
CD_mun = geography[['CD_MUN']]
# Step 2 — make a copy and truncate last digit of hfp['CD_MUN']
CD_mun['CD_MUN_6'] = CD_mun['CD_MUN'] // 10  # integer division removes last digit
# Step 3 — merge CD_MUN to hfp using 6-digit match
merged = hfp.merge(CD_mun, left_on='CD_MUN', right_on='CD_MUN_6', how='right', indicator=True)
# Step 4 — replace CD_MUN['CD_MUN'] with official 7-digit codes
merged = merged[['CD_MUN_y', 'CD_MUN_x', 'human_footprint', '_merge']]
# Step 5 — identify which entry is missing from CD_MUN
missing = merged.loc[merged['_merge'] == 'right_only', ['CD_MUN_y', 'CD_MUN_x', 'human_footprint']] # Lucena 2508604
# Step 6 - Fill missing municipality
merged.loc[merged['CD_MUN_y'] == missing['CD_MUN_y'].values[0], 'human_footprint'] = 25 # set to middle of scale
# Step 7 - Retain only desired columns
merged['CD_MUN'] = merged['CD_MUN_y']
hfp_copy = merged[['CD_MUN', 'human_footprint']]


# Aggregate to the intermediate/immediate regions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

for region, name in zip(regions, names):

    # input check
    assert ((region != 'CD_MUN') | (region != 'CD_RGI') |  (region != 'CD_RGINT')), "'region' must be either 'CD_MUN', 'CD_RGI' or 'CD_RGINT''"

    # Compute demographically weighted average
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if region != 'CD_MUN':

        # reload the raw data
        hfp = hfp_copy.copy()
        geography = gpd.read_parquet("../../data/interim/geographic-dataset.parquet")

        # get municipality to region mapping
        municipality_region_map = geography[['CD_MUN', f'{region}']]

        # get muncipality population
        municipality_demograhpy = geography[['CD_MUN', 'POP']]

        # Merge human footprint data with region map
        hfp = hfp.merge(municipality_region_map, on="CD_MUN", how="left")

        # Merge human footprint data with population counts
        hfp = hfp.merge(municipality_demograhpy, on="CD_MUN", how="left")
        
        # Compute population weighted index P
        hfp = (
            hfp
            .groupby([f'{region}'])
            .apply(lambda g: (g["human_footprint"] * g["POP"]).sum() / g["POP"].sum())
            .reset_index()
        )
        hfp = hfp.rename(columns={0: 'human_footprint'})

    else:
        hfp = hfp_copy.copy()

    # Visualise result
    # >>>>>>>>>>>>>>>>
    
    # Dissolve geometry and add human footprint
    dissolved_geography = geography.dissolve(by=f'{region}')
    dissolved_geography[f'human_footprint-{name}'] = hfp['human_footprint'].values

    # Make map
    fig,ax=plt.subplots()
    dissolved_geography.plot(
        column=dissolved_geography[f'human_footprint-{name}'],
        categorical=False,
        edgecolor=None,
        legend=True,
        ax=ax
    )
    ax.axis("off")
    ax.set_title("Human footprint")
    plt.tight_layout()
    plt.savefig(f'../interim/human-footprint/human-footprint-{name}.svg')
    plt.close()

    # Save result
    # >>>>>>>>>>>

    hfp.to_csv(f'../interim/human-footprint/human-footprint_{name}.csv', index=False)
