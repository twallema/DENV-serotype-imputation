import os
import ast
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

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
    if d1.year - d2.year > 0:
        return d1 # e.g. d2 is from previous calendar year --> use notification date to avoid dropping data (means first week of the year should have some artificial inflation)
    elif d2 < d1 - timedelta(days=30):
        return d1 # e.g. d2 is more than 30 days before d1 --> rare + something likely went wrong (date of birth often swapped with symptom onset, year mistaken)
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

# Loop over files
df_collect=[]
for fn,yr in zip(filenames, corresponding_years):

    # 1996, 1997, 1998
    if 1996 <= yr <= 1998:
        # define serotype column name
        serotype_column = 'SOROTIPO'
        # load data
        df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', delimiter=';')
        # find right Brazilian UF
        zip2uf_map = pd.read_csv('../raw/sprint_2025/map_regional_health.csv')[['uf', 'geocode']].drop_duplicates().set_index('geocode')['uf'].to_dict()
        df['SG_UF'] = df['MUNIATEND'].map(zip2uf_map)
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
        df = df.dropna(subset=['date'])
        # the column telling us if the case was 'confirmed' is unknown --> have to assume all cases are confirmed (even though 1999-2006 indicates this is not the case)
        
        pass

    elif 1999 <= yr <= 2006:
        # define serotype column name
        serotype_column = 'RESUL_VIRA'
        # load data
        read_csv_kwargs = {'delimiter': ';'} if yr == 1999 else {'delimiter':',', 'encoding':"ISO-8859-1"}
        df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', **read_csv_kwargs)
        # find most likely date
        ## strategy: take minimum of columns containing a date: ['DT_NOTIFIC', 'DT_SIN_PRI', 'DT_FEBRE'] # consider adding collection date
        ## Lags compared to 'DT_NOTIFIC':
        ## 1999: DT_SIN_PRI (-5.5, CL: -53, 0), DT_FEBRE (21, CL: -51, 72);
        ## 2000: DT_SIN_PRI (-6.8, CL: -55, 0), DT_FEBRE (17, CL: -27, 1089);
        ## 2001: DT_SIN_PRI (-7.7, CL: -59, 0), DT_FEBRE (-54, CL: -1271, 21);
        ## 2002: DT_SIN_PRI (-7.7, CL: -59, 0), DT_FEBRE (-54, CL: -1271, 21);
        ## 2003: DT_SIN_PRI (-12.2, CL: -54, 0), DT_FEBRE (-70, CL: -1592, 3);
        ## 2004: DT_SIN_PRI (-18.9, CL: -61, 0), DT_FEBRE (-9.1, CL: -63, 1);
        ## 2005: DT_SIN_PRI (-17.4, CL: -52, 0), DT_FEBRE (-10, CL: -62, 1);
        ## 2006: DT_SIN_PRI (-20.6, CL: -59, 0), DT_FEBRE (-10, CL: -63, 0);
        ## Medians are always -3/-4 days for both variables; IQR for DT_SIN_PRI is always in the range -7 --> -1
        ## DT_FEBRE = UNRELIABLE, average lag of DT_SIN_PRI is OK but I'm not transferring cases between years (this is probably an extensive recode)
        date_columns = ['DT_NOTIFIC', 'DT_SIN_PRI']
        df[date_columns] = df[date_columns].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')) # drop all errors 
        # find minimum date
        df['date'] = df[date_columns].apply(choose_date, axis=1)
        # drop if date not present (very rare)
        df = df.dropna(subset=['date'])
        # filter out the discards (CON_CLASSI==5) but leave in the undocumented value '0' and inconclusive '8'
        df = df[df['CON_CLASSI'] != 5]

        pass

    elif 2007 <= yr <= 2025:
        # define serotype column name
        serotype_column = 'SOROTIPO'
        # load data
        if yr == 2008:
            df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', delimiter=',', encoding="ISO-8859-1")
            # remove b'' for 2008 (using raw data)
            df = df.applymap(decode_or_nan)
            # convert 'SOROTIPO' and 'SG_UF' to numerics
            df['SOROTIPO'] = pd.to_numeric(df['SOROTIPO'])
            df['SG_UF'] = pd.to_numeric(df['SG_UF'])
            df['CLASSI_FIN'] = pd.to_numeric(df['CLASSI_FIN'])
        else:
            df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', delimiter=',', encoding="ISO-8859-1")
        # convert UF code in UF abbreviation
        zip2uf_map = pd.read_csv('../raw/sprint_2025/map_regional_health.csv')[['uf', 'uf_code']].drop_duplicates().set_index('uf_code')['uf'].to_dict()
        df['SG_UF'] = df['SG_UF'].map(zip2uf_map)
        # find most likely date
        ## strategy: take minimum of columns containing a date: ['DT_NOTIFIC', 'DT_SIN_PRI'] # consider adding collection date
        ## 2007: DT_SIN_PRI (-8.4, CL: -53, 0, IQR: -7, -2)
        ## 2008: DT_SIN_PRI (-21.8, CL: -68, 0, IQR: -7, -1)
        ## 2009: DT_SIN_PRI (-27.8, CL: -43, 0, IQR: -6, -1)
        ## 2010: DT_SIN_PRI (-25.1, CL: -49, 0, IQR: -6, -1)
        ## 2011: DT_SIN_PRI (-25.4, CL: -51, 0, IQR: -6, -1)
        ## Very similar to 1999-2006 
        date_columns = ['DT_NOTIFIC', 'DT_SIN_PRI']
        if yr == 2008:
            df[date_columns] = df[date_columns].apply(lambda x: pd.to_datetime(x, format='%Y%m%d', errors='coerce')) # drop all errors 
        else:
            df[date_columns] = df[date_columns].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')) # drop all errors 

        # Analyse lags
        # mean = {}
        # median = {}
        # q25 = {}
        # q75 = {}
        # q1 = {}
        # q99 = {}
        # for col in date_columns[1:]:  # Skip the reference column
        #     lag = (df[col] - df['DT_NOTIFIC']).dt.total_seconds() / (24 * 3600)  # convert to days
        #     mean[col] = lag.mean()
        #     median[col] = lag.median()
        #     q25[col] = lag.quantile(0.25)
        #     q75[col] = lag.quantile(0.75)
        #     q1[col] = lag.quantile(0.01)
        #     q99[col] = lag.quantile(0.99)
        # df = pd.DataFrame({'mean': mean, 'median': median, 'q25': q25, 'q75': q75, 'q1': q1, 'q99': q99})

        # find minimum date
        df['date'] = df[date_columns].apply(choose_date, axis=1)
        # drop if date is missing (very rare)
        df = df.dropna(subset=['date'])

        # Examination of cases that get a 'discarded' status after investigation BUT have a subtype assigned to them (pre 2008: CON_CLASSI)
        #print(df['CLASSI_FIN'].value_counts())
        #print(f'{yr}: Discarded status with serotype: {len(df[( (df['CLASSI_FIN'] == '5') & (~df['SOROTIPO'].isna()))])} out of {len(df[~df['SOROTIPO'].isna()])} serotyped entries')
        #print(f'{yr}: Discarded status with serotype: {len(df[( ((df['CLASSI_FIN'] == '5') | (df['CLASSI_FIN'] == '8')) & (~df['SOROTIPO'].isna()))])} out of {len(df[~df['SOROTIPO'].isna()])} serotyped entries')  
        
        # filter out 'discarded' (CLASSI_FIN==5) but keep 'inconclusive' (CLASSI_FIN==8)
        df = df[df['CLASSI_FIN'] != 5]

        pass

    # convert to the next saturday
    df['date'] = df['date'].apply(next_saturday)
    # clean the serotype column
    df['serotype'] = df[serotype_column].where(df[serotype_column].isin([1, 2, 3, 4]), np.nan)
    # retain only relevant columns
    df = df[['date', 'SG_UF', 'serotype']]
    # rename UF column for convenience
    df = df.rename(columns={'SG_UF': 'UF'})
    # drop if patient residency not provided 
    df = df.dropna(subset=['UF'])
    # build an expanded dataframe
    all_dates = pd.date_range(start=f'{yr}-01-01', end=f'{yr}-12-31', freq='W-SAT')
    all_states = pd.read_csv('../raw/sprint_2025/map_regional_health.csv')['uf'].unique()
    full_index = pd.MultiIndex.from_product([all_dates, all_states], names=['date', 'UF'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    # count serotypes
    serotype_counts = (
        df.dropna(subset=['serotype'])
        .groupby(['date', 'UF', 'serotype'])
        .size()
        .unstack(level='serotype')  # wide format, columns are 1.0–4.0
        .reindex(columns=[1.0, 2.0, 3.0, 4.0], fill_value=np.nan)  # ensures all 4 exist
        .rename(columns=lambda x: f'DENV_{int(x)}')
        .reset_index()
    )
    # count total observations
    total_counts = (
        df.groupby(['date', 'UF'])
        .size()
        .reset_index(name='DENV_total')
    )
    # merge together 
    final_df = (
        full_df
        .merge(serotype_counts, on=['date', 'UF'], how='left')
        .merge(total_counts, on=['date', 'UF'], how='left')
    )
    # save result
    df_collect.append(final_df)

# Final concatenation of dataframes
df = pd.concat(df_collect, ignore_index=True)
weekly_df = df.sort_values(by=['date', 'UF']).reset_index(drop=True)

# Save result (weekly frequency)
weekly_df.to_csv('../interim/datasus_DENV-linelist/uf/DENV-serotypes_1996-2025_weekly.csv', index=False)

# Save result (monthly frequency)
monthly_df = (
    df.set_index(['UF', 'date'])
    .groupby(level='UF')              # Group by state
    .resample('ME', level='date')     # Resample by month at the 'date' level
    .sum(min_count=1)                 # Ensure NaN if all values are NaN
    .reset_index()                    # Flatten index
)
monthly_df.to_csv('../interim/datasus_DENV-linelist/uf/DENV-serotypes_1996-2025_monthly.csv', index=False)

###################
## Visualisation ##
###################

# Visualise results 
## Brasil
df_vis = monthly_df.groupby(by='date').sum(min_count=1)
fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/1.5), sharex=True)
### Not serotyped
ax[0].plot(df_vis.index, df_vis.loc[slice(None), 'DENV_total'], color='black')
### Serotyped + serotyped zoom
for i in [1,2]:
    ax[i].plot(df_vis.index, df_vis.loc[slice(None), 'DENV_1'], color='black', label='DENV 1')
    ax[i].plot(df_vis.index, df_vis.loc[slice(None), 'DENV_2'], color='red', label='DENV 2')
    ax[i].plot(df_vis.index, df_vis.loc[slice(None), 'DENV_3'], color='green', label='DENV 3')
    ax[i].plot(df_vis.index, df_vis.loc[slice(None), 'DENV_4'], color='blue', label='DENV 4')
ax[2].legend()
### Axis decorations
ax[0].set_title('Brasil')
ax[0].set_ylabel('Monthly DENV incidence')
ax[1].set_ylabel('Monthly DENV incidence')
ax[2].set_ylabel('Monthly DENV incidence')
mx = max([max(df_vis.loc[slice(None), 'DENV_1'].values), max(df_vis.loc[slice(None), 'DENV_2'].values), max(df_vis.loc[slice(None), 'DENV_3'].values), max(df_vis.loc[slice(None), 'DENV_4'].values)])
ax[2].set_ylim([0, 0.06*mx])
plt.savefig('../interim/datasus_DENV-linelist/uf/figs/Brasil.png', dpi=300)
plt.close()

## States
df_vis = monthly_df.set_index(['date', 'UF'])
dates = df_vis.index.get_level_values('date').unique()
for UF in df_vis.index.get_level_values('UF').unique():
    fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/1.5), sharex=True)
    ### Not serotyped
    ax[0].plot(dates, df_vis.loc[(slice(None), UF), 'DENV_total'], color='black')
    ### Serotyped + serotyped zoom
    for i in [1,2]:
        ax[i].plot(dates, df_vis.loc[(slice(None), UF), 'DENV_1'], color='black', label='DENV 1')
        ax[i].plot(dates, df_vis.loc[(slice(None), UF), 'DENV_2'], color='red', label='DENV 2')
        ax[i].plot(dates, df_vis.loc[(slice(None), UF), 'DENV_3'], color='green', label='DENV 3')
        ax[i].plot(dates, df_vis.loc[(slice(None), UF), 'DENV_4'], color='blue', label='DENV 4')
    ### Axis decorations
    ax[1].legend()
    ax[0].set_ylabel('Monthly DENV incidence')
    ax[1].set_ylabel('Monthly DENV incidence')
    ax[2].set_ylabel('Monthly DENV incidence')
    ax[0].set_title(f'{UF}')
    ax[0].set_xlim([min(dates), max(dates)])
    mx = max([np.nanmax(df_vis.loc[(slice(None), UF), 'DENV_1'].values), np.nanmax(df_vis.loc[(slice(None), UF), 'DENV_2'].values), np.nanmax(df_vis.loc[(slice(None), UF), 'DENV_4'].values)])
    ax[2].set_ylim([0, 0.15*mx]) if not np.isnan(mx) else ax[2].set_ylim([0, 100])
    plt.savefig(f'../interim/datasus_DENV-linelist/uf/figs/{UF}.png', dpi=300)
    plt.close()