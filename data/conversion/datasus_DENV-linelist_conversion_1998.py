import os
import ast
import polars as pl
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import timedelta,date
import matplotlib.pyplot as plt

start_year = 1998
end_year = 2026

assert start_year == 1998, 'start_year must be 1998'

######################
## Helper functions ##
######################

# Define a function that maps any date to the next Saturday (end of CDC epiweek is always a Saturday)
def next_saturday(date):
    if pd.isna(date):
        return pd.NaT
    days_ahead = (5 - date.weekday()) % 7  # 5 = Saturday (Monday=0)
    days_ahead = 7 if days_ahead == 0 else days_ahead  # skip to next if already Saturday
    return date + pd.Timedelta(days=days_ahead)

# 2008: Clean all entries: if it's "b' '" → NaN, else decode the content
def decode_or_nan(x):
    if isinstance(x, str) and x.strip() == "b' '":
        return np.nan
    elif isinstance(x, str) and x.strip() == "b'  '":
        return np.nan
    elif isinstance(x, str) and x.strip() == "b'      '":
        return np.nan
    try:
        val = ast.literal_eval(x)
        return val.decode('utf-8') if isinstance(val, bytes) else val
    except (ValueError, SyntaxError):
        return x  # fallback in case it's not a byte string

# Custom selection logic
def choose_date(row):
    # unpack dates (notification vs. sypmtom onset)
    d1, d2 = row['DT_NOTIFIC'], row['DT_SIN_PRI']
    # logic
    if d2 < d1 - timedelta(days=60):
        return d1
    else:
        return d2

####################
## Data wrangling ##
####################

# Load and sort all filenames
filenames = [f for f in os.listdir('../raw/datasus_DENV-linelist/composite_dataset') if os.path.isfile(os.path.join('../raw/datasus_DENV-linelist/composite_dataset', f)) and f != '.DS_Store' and f != 'README.md' and f != '.Rhistory']
filenames.sort()

# Figure out corresponding year
corresponding_years = [int(fn[10:14]) for fn in filenames]
extracted_years = [yr for yr in corresponding_years if ((yr >= start_year) & (yr <= end_year))]
extracted_years_mask = [True if yr in extracted_years else False for yr in corresponding_years]
filenames = [fn for i,fn in enumerate(filenames) if extracted_years_mask[i] == True]

# Formatted data collection
df_muni_age_collect=[]

# Get the municipality to federative unit map, municipality to immediate region map, and immediate region to federative unit map
mun2uf_map = gpd.read_parquet('../interim/geographic-dataset.parquet')[['CD_UF', 'CD_MUN']].drop_duplicates().set_index('CD_MUN')['CD_UF'].to_dict()
code2name_uf_map = gpd.read_parquet('../interim/geographic-dataset.parquet')[['CD_UF', 'NM_UF']].drop_duplicates().set_index('CD_UF')['NM_UF'].to_dict()



# Extract data
# >>>>>>>>>>>>

fn = filenames[0]
yr = extracted_years[0]

# define serotype column name
serotype_column = 'SOROTIPO'
# load data
df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', delimiter=';', low_memory=False)
# rename municipality geocode column for consistency
df = df.rename(columns={'MUNIATEND': 'CD_MUN'})
# attach relevant spatial units
df['CD_UF'] = df['CD_MUN'].map(mun2uf_map)
# find most likely date
## strategy: take minimum of columns containing a date: ['DTCOLETA', 'DTMAC1', 'DTMAC2', 'DTINIHEMA1', 'DTINIHEMA2']
## BUT: MAC1/MAC2/DTINIHEMA1/DTINIHEMA2 always lag 'DTCOLECTA' (98% confidence interval > 0), except MAC1 in 1997 which has strongly negative lagging outliers compared to 'DTCOLECTA' (1% lags more than 150 days)
## HENCE: use 'DTCOLECTA' only 
## BUT: there are a lot of missing dates so we wind up missing out on a lot of data by only using 'DTCOLECTA'
date_columns = ['DTCOLETA', 'DTMAC1', 'DTMAC2', 'DTINIHEMA1', 'DTINIHEMA2']
df[date_columns] = df[date_columns].apply(pd.to_datetime)
# find minimum date
df['date'] = df[date_columns].min(axis=1)
# drop if date not present (very rare)
print(f"Fraction with a missing date: {100 - len(df.dropna(subset=['date'])) / len(df) * 100} %")
df = df.dropna(subset=['date'])
# the column telling us if the case was 'confirmed' is unknown --> assume no cases are confirmed, except the serotyped ones
df['confirmed'] = df[serotype_column].isin([1, 2, 3, 4]).astype(int)


# General conversions 
# >>>>>>>>>>>>>>>>>>>

# convert to the next saturday
df['date'] = df['date'].apply(next_saturday)
# clean the serotype column
df['serotype'] = df[serotype_column].where(df[serotype_column].isin([1, 2, 3, 4]), pd.NA)
# set minimal data types
df = df.astype({'serotype': 'Int8', 'CD_MUN': 'Int32'})

# Convert linelist data into counts
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print("\nCounting the data..")

group_cols = ["date", "CD_MUN"]

# retain only relevant columns
df = df[["date", "CD_MUN", "serotype"]]

# convert to polars
df = pl.from_pandas(df)

# sort
df = df.sort(by=group_cols+['serotype',])

# convert dates to month end
df = df.with_columns(pl.col("date").dt.month_end())

# Total counts
total_counts = (
    df
    .group_by(group_cols)
    .len()
    .rename({"len": "DENV_total"})
)

# Serotype counts
serotype_counts = (
    df
    .filter(pl.col("serotype").is_not_null())
    .group_by(group_cols + ["serotype"])
    .len()
    .pivot(
        index=group_cols,
        on="serotype",
        values="len",
    )
)

# ensure all serotype columns exist
for s in [1, 2, 3, 4]:
    if str(s) not in serotype_counts.columns:
        serotype_counts = serotype_counts.with_columns(
            pl.lit(None).cast(pl.Int32).alias(str(s))
        )

serotype_counts = serotype_counts.rename({
    "1": "DENV_1",
    "2": "DENV_2",
    "3": "DENV_3",
    "4": "DENV_4",
})

# Build Cartesian product dataframe
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

all_months = pl.datetime_range(date(yr,1,31), date(yr,12,31), interval="1mo", eager=True).dt.month_end()
months = pl.DataFrame({"date": all_months})

all_muni = pl.from_pandas(pd.Series(gpd.read_parquet('../interim/geographic-dataset.parquet')['CD_MUN'].unique(), name='CD_MUN')).to_frame()

full_df = (
    months
    .lazy()
    .join(all_muni.lazy(), how="cross")
)

# Merge counts
df_1998 = (
    full_df
    .join(
        total_counts.lazy(),
        on=group_cols,
        how="left",
    )
    .join(
        serotype_counts.lazy(),
        on=group_cols,
        how="left",
    )
    .sort(group_cols)
    .collect(engine="streaming")
)

# Merge to the dataframe without age or diagnostics
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print("\nMerging to the existing data without age or diagnostics..")

# get the complete dataset
df = pl.scan_parquet(f"../interim/datasus_DENV-linelist/DENV-{start_year+1}_{end_year}-month-mun-no_diagnostics.parquet").collect(engine="streaming")

# merge the 1998 data with the 1998-2026 data and sum duplicates (preserving null)
data_cols = ["DENV_total", "DENV_1", "DENV_2", "DENV_3", "DENV_4"]

df_merged = (
    pl.concat(
        [
            df.select(["date", "CD_MUN"] + data_cols),
            df_1998.select(["date", "CD_MUN"] + data_cols),
        ],
        how="vertical",
    )
    .group_by(["date", "CD_MUN"])
    .agg([
        pl.when(pl.col(col).is_not_null().any())
        .then(pl.col(col).sum())
        .otherwise(None)
        .alias(col)
        for col in data_cols
    ])
    .sort(["date", "CD_MUN"])
)

# save output
df_merged.write_parquet(
    "../interim/datasus_DENV-linelist/"
    f"DENV-{start_year}_{end_year}-month-mun-age_group-no_diagnostics.parquet",
    compression="zstd",
)


#############################
## Visualisation (UF only) ##
#############################

print("\nMaking visualisations..")

# groupby-sum to UF level
df_muni = df_merged.to_pandas()
df_muni['CD_UF'] = df_muni["CD_MUN"].astype(str).str[:2]
cols_to_sum = ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4', 'DENV_total']
df_uf = df_muni.groupby(by=['date', 'CD_UF'], dropna=False)[cols_to_sum].sum(min_count=1).reset_index()

# Visualise results 
## Brasil
fig,ax=plt.subplots(nrows=2, figsize=(8.3,11.7/2), sharex=True)
### Not serotyped
df_vis = df_uf.groupby(by=['date'], dropna=False)[cols_to_sum].sum(min_count=1).reset_index()
x = df_vis.date.unique()
ax[0].plot(x, df_vis['DENV_total'], color='black')
ax[0].set_ylabel('Monthly number of dengue cases (-)')
ax[0].legend()
### Serotyped
df_vis = df_vis.groupby(by='date')[cols_to_sum].sum()
ax[1].plot(x, df_vis['DENV_1'], color='black', label='DENV 1')
ax[1].plot(x, df_vis['DENV_2'], color='red', label='DENV 2')
ax[1].plot(x, df_vis['DENV_3'], color='green', label='DENV 3')
ax[1].plot(x, df_vis['DENV_4'], color='blue', label='DENV 4')
ax[1].set_ylabel('Monthly serotyped cases (-)')
ax[1].legend(loc=1)
### Axis decorations
ax[1].set_title('Brasil')
os.makedirs('../interim/datasus_DENV-linelist/figs/including_2008', exist_ok=True)
plt.savefig('../interim/datasus_DENV-linelist/figs/including_2008/Brasil.png', dpi=300)
plt.close()

## States
x = df_uf.date.unique()
for UF in df_uf.CD_UF.unique():
    fig,ax=plt.subplots(nrows=2, figsize=(8.3,11.7/2), sharex=True)
    ### Not serotyped
    df_vis = df_uf[df_uf['CD_UF'] == UF].reset_index()
    ax[0].plot(x, df_vis['DENV_total'], color='black')
    ax[0].set_ylabel('Monthly DENV incidence')
    ax[0].legend()
    ### Serotyped + serotyped zoom
    ax[1].plot(x, df_vis['DENV_1'], color='black', label='DENV 1')
    ax[1].plot(x, df_vis['DENV_2'], color='red', label='DENV 2')
    ax[1].plot(x, df_vis['DENV_3'], color='green', label='DENV 3')
    ax[1].plot(x, df_vis['DENV_4'], color='blue', label='DENV 4')
    ax[1].set_ylabel('Monthly DENV cases')
    ax[1].legend(loc=1)
    ### Axis decorations
    ax[0].set_ylabel('Monthly DENV incidence')
    ax[0].set_title(f"{code2name_uf_map[int(UF)]}")
    plt.savefig(f'../interim/datasus_DENV-linelist/figs/including_2008/{code2name_uf_map[int(UF)]}.png', dpi=300)
    plt.close()