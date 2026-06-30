import os
import ast
import random
import pandas as pd
import numpy as np
import geopandas as gpd
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
    if d1.year - d2.year > 0:
        return d1 # e.g. d2 is from previous calendar year --> use notification date to avoid dropping data (means first week of the year should have some artificial inflation)
    elif d2 < d1 - timedelta(days=30):
        return d1 # e.g. d2 is more than 30 days before d1 --> rare + something likely went wrong (date of birth often swapped with symptom onset, year mistaken)
    else:
        return d2

# parse age codes
## handle both numeric (post 2007) and alphanumeric (pre 2007) formats
def parse_age(x):
    if pd.isna(x):
        return pd.NA

    x = str(x).strip()

    # 1-2 digits --> assume it's years
    if x.isdigit() and len(x) <= 2:
        return int(x)

    # Numeric encoding (1005, 3002, 4064, ...)
    if x.isdigit() and len(x) == 4:
        unit = x[0]
        value = int(x[1:])

    # Alphabetic encoding (A025, M003, D005, H023)
    elif len(x) == 4 and x[0] in "AMDHS":
        unit = x[0]
        value = int(x[1:])

    else:
        return pd.NA

    if unit in ("1", "2", "H", "D", "S"):
        return 0
    elif unit in ("3", "M"):
        return value // 12
    elif unit in ("4", "A"):
        return value
    else:
        return pd.NA


####################
## Data wrangling ##
####################

# Load and sort all filenames
filenames = [f for f in os.listdir('../raw/datasus_DENV-linelist/composite_dataset') if os.path.isfile(os.path.join('../raw/datasus_DENV-linelist/composite_dataset', f)) and f != '.DS_Store' and f != 'README.md' and f != '.Rhistory']
filenames.sort()

# Figure out corresponding year
corresponding_years = [int(fn[10:14]) for fn in filenames]

# Formatted data collection
df_uf_collect=[]
df_muni_collect=[]
df_muni_age_collect=[]

# Get the municipality to federative unit map, municipality to immediate region map, and immediate region to federative unit map
mun2uf_map = gpd.read_parquet('../interim/geographic-dataset.parquet')[['CD_UF', 'CD_MUN']].drop_duplicates().set_index('CD_MUN')['CD_UF'].to_dict()

# Loop over files
for fn,yr in zip(filenames[3:12], corresponding_years[3:12]):
    print(f'\nWorking on year {yr}')
    print('---------------------')
    print("\nWorking on preprocessing..")
    # 1996, 1997, 1998
    if 1996 <= yr <= 1998:
        raise ValueError("script no longer works for years before 1999.\n")
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
        pass

    elif 1999 <= yr <= 2006:
        # define relevant column names
        serotype_column = 'RESUL_VIRA'
        age_column = 'NU_IDADE'
        classification_column = 'CON_CLASSI'
        criterion_column = 'CON_CRITER'
        # load data
        read_csv_kwargs = {'delimiter': ';'} if yr == 1999 else {'delimiter':',', 'encoding':"ISO-8859-1"}
        df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', **read_csv_kwargs, low_memory=False)

        # find most likely date
        # DT_COLLECTA very reliable according to Laura
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
        df[date_columns] = df[date_columns].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')) # errors --> NaT
        # find minimum date
        df['date'] = df[date_columns].apply(choose_date, axis=1)
        # drop if date not present (very rare)
        print(f"[DROPPED] Fraction with a missing/invalid date: {100 - len(df.dropna(subset=['date'])) / len(df) * 100:.2f} %")
        df = df.dropna(subset=['date'])

        # drop if age is missing
        print(f"[DROPPED] Fraction with a missing age: {100 - len(df.dropna(subset=[age_column])) / len(df) * 100} %")
        df = df.dropna(subset=[age_column])
        # convert NU_IDADE to age
        df = df.rename(columns={age_column: 'age'})
        # convert age code to an age in years
        l = len(df)
        df["age"] = df["age"].apply(parse_age).astype("Int64")
        print(f"[DROPPED] Fraction with an invalid age: {100 - len(df.dropna(subset=['age'])) / l * 100} %")
        df = df.dropna(subset=['age'])

        # [SANITY CHECK] fraction with a serotype assigned but a missing classification
        print(f"[FYI] Fraction with serotype information but missing a classification: {100 - len(df.dropna(subset=serotype_column).dropna(subset=classification_column)) / len(df.dropna(subset=[serotype_column])) * 100:.2f} %")
        # drop if missing classification
        print(f"[DROPPED] Fraction with a missing classification: {100 - len(df.dropna(subset=[classification_column])) / len(df) * 100:.2f} %")
        df = df.dropna(subset=[classification_column])

        # how much of each classification are there
        print(f"[FYI] Distribution of classifications (%): \n")
        print(df[classification_column].value_counts() / sum(df[classification_column].value_counts()) * 100)
        print('\n')
        print(f"[DROPPED] undocumented classification '0'")
        # drop if classification is "0" (undocumented; only in 1999-2003)
        df = df[df[classification_column] != 0]

        # fill in discarded
        df['discarded'] = df[classification_column].isin([5,]).astype(int)
        # fill diagnosis code
        print(f"[FYI] Fraction with reason for classification missing (assigned to unknown): {len(df[df[criterion_column].isna()]) / len(df) * 100:.2f} %, of which {len(df[((df[criterion_column].isna()) & (df[classification_column] == 5))]) / len(df) * 100:.2f} % discarded")
        print(f"[FYI] Fraction with reason for classification listed as 'under investigation' (assigned to unknown): {len(df[df[criterion_column] == 3]) / len(df) * 100:.2f} %")
        df['diagnosis'] = df[criterion_column].fillna('unknown')
        df.loc[df['diagnosis']==1, 'diagnosis'] = 'lab'
        df.loc[df['diagnosis']==2, 'diagnosis'] = 'clin-epi'
        df.loc[df['diagnosis']==3, 'diagnosis'] = 'unknown'

        # fill in severity (NA if discarded  (5), 'inconclusive' if inconclusive (8))
        df['severity'] = 'NA'
        df.loc[df[classification_column]==1, 'severity'] = 'low'
        df.loc[df[classification_column]==2, 'severity'] = 'medium'
        df.loc[((df[classification_column]==3) | (df[classification_column]==4)), 'severity'] = 'high'
        df.loc[df[classification_column]==8, 'severity'] = 'inconclusive'
        print(f"[FYI] Classification 'discard' (5) assigned severity 'NA'")
        print(f"[FYI] Classification 'inconclusive' (8) assigned severity 'inconclusive'")
        print(f"[FYI] Unique severities when discard==FALSE: {df[df['discarded'] == 0]['severity'].unique()}")

        # Rename ID_MN_RES to CD_MUN
        df = df.rename(columns={'ID_MN_RESI': 'CD_MUN'})
        # drop if patient residency not provided
        print(f"[DROPPED] Fraction with missing municipality code: {100 - len(df.dropna(subset=['CD_MUN'])) / len(df) * 100:.2f} %")
        df = df.dropna(subset=['CD_MUN'])

        pass

    elif 2007 <= yr <= 2025:
        # define serotype column name
        serotype_column = 'SOROTIPO'
        age_column = 'NU_IDADE_N'
        classification_column = 'CLASSI_FIN'
        criterion_column = 'CRITERIO'
        # load data
        if yr == 2008:
            df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', delimiter=',', encoding="ISO-8859-1", low_memory=False)
            # remove b'' for 2008 (using raw data)
            df = df.map(decode_or_nan)
            # convert 'SOROTIPO' and 'SG_UF' to numerics
            df[serotype_column] = pd.to_numeric(df[serotype_column])
            df['ID_MN_RESI'] = pd.to_numeric(df['ID_MN_RESI'], errors='coerce') # Has 6 digits!
            df[classification_column] = pd.to_numeric(df[classification_column])
        else:
            df = pd.read_csv(f'../raw/datasus_DENV-linelist/composite_dataset/{fn}', delimiter=',', encoding="ISO-8859-1", low_memory=False)

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
        # find minimum date
        df['date'] = df[date_columns].apply(choose_date, axis=1)
        # drop if date is missing (very rare)
        print(f"[DROPPED] Fraction with a missing/invalid date: {100 - len(df.dropna(subset=['date'])) / len(df) * 100} %")
        df = df.dropna(subset=['date'])

        # drop if age is missing
        print(f"[DROPPED] Fraction with a missing age: {100 - len(df.dropna(subset=[age_column])) / len(df) * 100} %")
        df = df.dropna(subset=[age_column])
        # convert NU_IDADE to age
        df = df.rename(columns={age_column: 'age'})
        # convert age code to an age in years
        l = len(df)
        df["age"] = df["age"].apply(parse_age).astype("Int64")
        print(f"[DROPPED] Fraction with an invalid age: {100 - len(df.dropna(subset=['age'])) / l * 100} %")
        df = df.dropna(subset=['age'])

        # Rename ID_MN_RES to CD_MUN
        df = df.rename(columns={'ID_MN_RESI': 'CD_MUN'})
        # drop if patient residency not provided
        print(f"[DROPPED] Fraction with missing municipality code: {100 - len(df.dropna(subset=['CD_MUN'])) / len(df) * 100:.2f} %")
        df = df.dropna(subset=['CD_MUN'])
        # Salvage the last m*therf*cking digit @!!*@\
        df['CD_MUN'] = df['CD_MUN'].astype(int)
        valid_codes = list(mun2uf_map.keys())
        # Step 1: build lookup dictionary: {six_digit: [seven_digit candidates]}
        lookup = {}
        for code in valid_codes:
            six_digit = code // 10  # drop last digit
            lookup.setdefault(six_digit, []).append(code)
        # Step 2: map with warnings
        no_hits = 0
        non_unique_hits = 0
        mapped_codes = []
        for six_code in df["CD_MUN"]:
            candidates = lookup.get(six_code, [])
            if len(candidates) == 0:
                no_hits += 1
                #print(f"WARNING: no match for {six_code}")
                mapped_codes.append(None)  # or np.nan
            elif len(candidates) == 1:
                mapped_codes.append(candidates[0])
            else:
                non_unique_hits += 1
                choice = random.choice(candidates)
                #print(f"WARNING: multiple matches for {six_code} → {candidates}, picked {choice}")
                mapped_codes.append(choice)
        df["CD_MUN_7"] = mapped_codes
        # Step 3: report fraction of ambiguous matches
        fraction_non_unique = non_unique_hits / len(df) 
        fraction_no_hits = no_hits / len(df)
        print(f"[FYI] Fraction of 6-digit municipality code with non-unique 7-digit matches (picked one at random): {fraction_non_unique:.2%}")
        print(f"[DROPPED] Fraction of 6-digit municipality code with no 7-digit matches: {fraction_no_hits:.2%}")
        # Step 4: drop nan and convert to integer
        df = df.dropna(subset='CD_MUN_7')
        df['CD_MUN'] = df['CD_MUN_7'].astype(int)

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


        # [SANITY CHECK] fraction with a serotype assigned but a missing outcome 
        print(f"[FYI] Fraction with serotype information but missing a classification: {100 - len(df.dropna(subset=serotype_column).dropna(subset=classification_column)) / len(df.dropna(subset=[serotype_column])) * 100:.2f} %")
        # drop if missing classification
        print(f"[DROPPED] Fraction with a missing classification: {100 - len(df.dropna(subset=[classification_column])) / len(df) * 100:.2f} %")
        df = df.dropna(subset=[classification_column])

        # how much of each classification are there
        print(f"[FYI] Distribution of classifications (%): \n")
        print(df[classification_column].value_counts() / sum(df[classification_column].value_counts()) * 100)
        print('\n')
        print(f"[DROPPED] undocumented classification '0'")
        # drop if classification is "0" (undocumented; only in 1999-2003)
        df = df[df[classification_column] != 0]

        # fill in discarded
        df['discarded'] = df[classification_column].isin([5,]).astype(int)
        # fill diagnosis code
        print(f"[FYI] Fraction with reason for classification missing (assigned to unknown): {len(df[df[criterion_column].isna()]) / len(df) * 100:.2f} %, of which {len(df[((df[criterion_column].isna()) & (df[classification_column] == 5))]) / len(df) * 100:.2f} % discarded")
        print(f"[FYI] Fraction with reason for classification listed as 'under investigation' (assigned to unknown): {len(df[df[criterion_column] == 3]) / len(df) * 100:.2f} %")
        df['diagnosis'] = df[criterion_column].fillna('unknown')
        df.loc[df['diagnosis']==1, 'diagnosis'] = 'lab'
        df.loc[df['diagnosis']==2, 'diagnosis'] = 'clin-epi'
        df.loc[df['diagnosis']==3, 'diagnosis'] = 'unknown'

        # fill in severity (NA if discarded  (5), 'inconclusive' if inconclusive (8))
        df['severity'] = 'NA'
        df.loc[df[classification_column]==1, 'severity'] = 'low'
        df.loc[df[classification_column]==2, 'severity'] = 'medium'
        df.loc[((df[classification_column]==3) | (df[classification_column]==4)), 'severity'] = 'high'
        df.loc[df[classification_column]==8, 'severity'] = 'inconclusive'
        print(f"[FYI] Classification 'discard' (5) assigned severity 'NA'")
        print(f"[FYI] Classification 'inconclusive' (8) assigned severity 'inconclusive'")
        print(f"[FYI] Unique severities when discard==FALSE: {df[df['discarded'] == 0]['severity'].unique()}")

        # Rename ID_MN_RES to CD_MUN
        df = df.rename(columns={'ID_MN_RESI': 'CD_MUN'})

        pass
    

    # General conversions 
    # >>>>>>>>>>>>>>>>>>>

    # convert to the next saturday
    df['date'] = df['date'].apply(next_saturday)
    # clean the serotype column
    df['serotype'] = df[serotype_column].where(df[serotype_column].isin([1, 2, 3, 4]), np.nan)


    # Collect serotype data at municipality level
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    print("\nWorking on municipality data collection..")
    df_copy = df.copy(deep=True)
    # retain only relevant columns
    df = df[['date', 'CD_MUN', 'discarded', 'diagnosis', 'severity', 'serotype']]
    # build an expanded dataframe
    all_dates = pd.date_range(start=f'{yr-1}-12-17', end=f'{yr+1}-01-14', freq='W-SAT')
    all_muni = gpd.read_parquet('../interim/geographic-dataset.parquet')['CD_MUN'].unique()
    full_index = pd.MultiIndex.from_product([all_dates, all_muni, [0,1], ['lab', 'clin-epi', 'unknown'], ['low', 'medium', 'high', 'NA']], names=['date', 'CD_MUN', 'discarded', 'diagnosis', 'severity'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    # count serotypes
    serotype_counts = (
        df.dropna(subset=['serotype'])
        .groupby(['date', 'CD_MUN', 'serotype', 'discarded', 'diagnosis', 'severity'])
        .size()
        .unstack(level='serotype')  # wide format, columns are 1.0–4.0
        .reindex(columns=[1.0, 2.0, 3.0, 4.0], fill_value=np.nan)  # ensures all 4 exist
        .rename(columns=lambda x: f'DENV_{int(x)}')
        .reset_index()
    )
    # count total observations
    total_counts = (
        df.groupby(['date', 'CD_MUN', 'discarded', 'diagnosis', 'severity'])
        .size()
        .reset_index(name='DENV_total')
    )
    # merge together 
    final_df = (
        full_df
        .merge(serotype_counts, on=['date', 'CD_MUN', 'discarded', 'diagnosis', 'severity'], how='left')
        .merge(total_counts, on=['date', 'CD_MUN', 'discarded', 'diagnosis', 'severity'], how='left')
    )
    # save result
    df_muni_collect.append(final_df)


    # Collect age-structured data at municipality level
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # print("\nWorking on municipality age-structured data collection..")
    # df = df_copy
    # if yr >= 1999:
    #     # retain only relevant columns
    #     df = df[['date', 'age', 'CD_MUN', 'discarded', 'diagnosis', 'serotype']]
    #     # drop if patient residency not provided
    #     print(f"Fraction with missing municipality code: {100 - len(df.dropna(subset=['CD_MUN'])) / len(df) * 100:.2f} %")
    #     df = df.dropna(subset=['CD_MUN'])
    #     # drop if patient age not provided
    #     l=len(df)
    #     df = df.dropna(subset=['age'])
    #     print(f"Fraction with missing age: {100 - len(df) / l * 100:.2f} %") 
    #     # limit age
    #     print(f"Fraction with 0 <= age <= 100: {len(df[((df['age'] >= 0) & (df['age'] <= 100))]) / len(df) * 100:.2f} %")
    #     df = df[((df['age'] >= 0) & (df['age'] <= 100))]

    #     # build an expanded dataframe
    #     all_dates = pd.date_range(start=f'{yr-1}-12-24', end=f'{yr+1}-01-07', freq='W-SAT')
    #     all_ages = np.arange(101)
    #     all_muni = gpd.read_parquet('../interim/geographic-dataset.parquet')['CD_MUN'].unique()
    #     full_index = pd.MultiIndex.from_product([all_dates, all_ages, all_muni, [0,1], ['lab', 'clin-epi', 'unknown']], names=['date', 'age', 'CD_MUN', 'discarded', 'diagnosis'])
    #     full_df = pd.DataFrame(index=full_index).reset_index()
        
    #     # count total observations
    #     total_counts = (
    #         df.groupby(['date', 'age', 'CD_MUN', 'discarded', 'diagnosis'])
    #         .size()
    #         .reset_index(name='DENV_total')
    #     )
    #     # merge together 
    #     df = (
    #         full_df
    #         .merge(total_counts, on=['date', 'age', 'CD_MUN', 'discarded', 'diagnosis'], how='left')
    #     )
    #     # bin age groups
    #     df['DENV_total'] = df['DENV_total'].fillna(0)
    #     bins = np.arange(0, 105, 5) 
    #     labels = [f"[{i:02d}-{i+5:02d}(" for i in range(0, 100, 5)]
    #     df['age_group'] = pd.cut(
    #         df['age'],
    #         bins=bins,
    #         right=False,   # intervals like [0,5)
    #         labels=labels,
    #         include_lowest=True
    #     )
    #     df_binned = (
    #         df
    #         .groupby(['date', 'CD_MUN', 'age_group', 'discarded', 'diagnosis'], as_index=False, observed=False)['DENV_total']
    #         .sum()
    #     )
    #     # save result
    #     df_muni_age_collect.append(df_binned)


# Final concatenation of dataframes at municipality spatial level
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# place them one after the other
df_muni = pd.concat(df_muni_collect, ignore_index=True)

# get rid of the overlapping week 
agg_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]
# Sum, treating NaNs as 0
summed = df_muni.groupby(["date", "CD_MUN", "discarded", "diagnosis", "severity"], as_index=False)[agg_cols].sum()
# Count non-missing values
counts = df_muni.groupby(["date", "CD_MUN", "discarded", "diagnosis", "severity"])[agg_cols].count()
# Restore NaN where all entries were NaN
summed[agg_cols] = summed[agg_cols].mask(counts.eq(0).values)
df_muni = summed

# group
weekly_df_muni = df_muni.sort_values(by=['date', 'CD_MUN']).reset_index(drop=True)

# Save result (weekly frequency)
weekly_df_muni.to_parquet('../interim/datasus_DENV-linelist/mun/DENV-serotypes_1999-2025_weekly_mun.parquet.gz', index=False, compression='gzip')

# Save result (monthly frequency)
monthly_df_muni = (
    df_muni.set_index(['CD_MUN', 'date', 'discarded', 'diagnosis', 'severity'])
    .groupby(level=['CD_MUN', 'discarded', 'diagnosis', 'severity'])                # Group by municipality
    .resample('ME', level='date')           # Resample by month at the 'date' level
    .sum(min_count=1)                       # Ensure NaN if all values are NaN
    .reset_index()                          # Flatten index
)
monthly_df_muni.to_parquet('../interim/datasus_DENV-linelist/mun/DENV-serotypes_1999-2025_monthly_mun.parquet.gz', index=False, compression='gzip')


# Final concatenation of age-structured dataframes at municipality spatial level
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# # place them one after the other
# df_muni_age = pd.concat(df_muni_age_collect, ignore_index=True)

# # get rid of the overlapping week 
# agg_cols = ["DENV_total",]
# # Sum, treating NaNs as 0
# summed = df_muni_age.groupby(["date", "CD_MUN", "age_group", "discarded", "diagnosis"], as_index=False)[agg_cols].sum()
# # Count non-missing values
# counts = df_muni_age.groupby(["date", "CD_MUN", "age_group", "discarded", "diagnosis"])[agg_cols].count()
# # Restore NaN where all entries were NaN
# summed[agg_cols] = summed[agg_cols].mask(counts.eq(0).values)
# df_muni_age = summed

# # group
# weekly_df_muni_age = df_muni_age.sort_values(by=['date', 'CD_MUN']).reset_index(drop=True)

# # Save result (monthly frequency only)
# monthly_df_muni_age = (
#     df_muni_age.set_index(['date', 'CD_MUN', 'age_group', 'discarded', 'diagnosis'])
#     .groupby(level=['CD_MUN', 'age_group', 'discarded', 'diagnosis'], observed=False)  # Group by state
#     .resample('ME', level='date')     # Resample by month at the 'date' level
#     .sum(min_count=1)                 # Ensure NaN if all values are NaN
#     .reset_index()                    # Flatten index
# )
# monthly_df_muni_age = monthly_df_muni_age.sort_values(by=['date', 'age_group', 'CD_MUN']).reset_index(drop=True)
# monthly_df_muni_age.to_parquet('../interim/datasus_DENV-linelist/mun/DENV_total_age_1999-2025_monthly_mun.parquet.gz', index=False, compression='gzip')

# print(monthly_df_muni_age.head(50))
# print(monthly_df_muni_age.tail(50))


#############################
## Visualisation (UF only) ##
#############################

# add UF label
monthly_df_muni['CD_UF'] = monthly_df_muni["CD_MUN"].astype(str).str[:2]
# selected non-discarded cases
monthly_df_muni = monthly_df_muni[monthly_df_muni['discarded'] == 0]
# groupby sum
monthly_df_uf = monthly_df_muni.groupby(by=['date', 'CD_UF', 'diagnosis'])[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4', 'DENV_total']].sum().reset_index()


# Visualise results 
## Brasil
fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/1.5), sharex=True)
### Not serotyped (by diagnosis)
df_vis = monthly_df_uf.groupby(by=['date', 'diagnosis']).sum(min_count=1).reset_index()
x = df_vis.date.unique()
ax[0].plot(x, df_vis[df_vis['diagnosis'] == 'clin-epi']['DENV_total'], color='red', label='clin/epi')
ax[0].plot(x, df_vis[df_vis['diagnosis'] == 'lab']['DENV_total'], color='green', label='lab')
ax[0].plot(x, df_vis[df_vis['diagnosis'] == 'unknown']['DENV_total'], color='blue', label='unknown')
ax[0].plot(x, df_vis.groupby(by='date')['DENV_total'].sum(), color='black', label='all')
ax[0].set_ylabel('Monthly DENV cases')
ax[0].legend()
### Not serotyped (by severity)


### Serotyped + serotyped zoom
for i in [1,2]:
    df_vis = df_vis.groupby(by='date')[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].sum()
    ax[i].plot(x, df_vis['DENV_1'], color='black', label='DENV 1')
    ax[i].plot(x, df_vis['DENV_2'], color='red', label='DENV 2')
    ax[i].plot(x, df_vis['DENV_3'], color='green', label='DENV 3')
    ax[i].plot(x, df_vis['DENV_4'], color='blue', label='DENV 4')
    ax[i].set_ylabel('Monthly DENV cases')
ax[2].legend(loc=1)
### Axis decorations
ax[0].set_title('Brasil')
ax[2].set_ylim([0, 100])
os.makedirs('../interim/datasus_DENV-linelist/figs', exist_ok=True)
plt.savefig('../interim/datasus_DENV-linelist/figs/Brasil.png', dpi=300)
plt.close()

## States
x = monthly_df_uf.date.unique()
for UF in monthly_df_uf.CD_UF.unique():
    fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/1.5), sharex=True)
    ### Not serotyped
    df_vis = monthly_df_uf[monthly_df_uf['CD_UF'] == UF]
    ax[0].plot(x, df_vis[df_vis['diagnosis'] == 'clin-epi']['DENV_total'], color='red', label='clin/epi')
    ax[0].plot(x, df_vis[df_vis['diagnosis'] == 'lab']['DENV_total'], color='green', label='lab')
    ax[0].plot(x, df_vis[df_vis['diagnosis'] == 'unknown']['DENV_total'], color='blue', label='unknown')
    ax[0].plot(x, df_vis.groupby(by='date')['DENV_total'].sum(), color='black', label='all')
    ax[0].set_ylabel('Monthly DENV incidence')
    ax[0].legend()
    ### Serotyped + serotyped zoom
    for i in [1,2]:
        df_vis = df_vis.groupby(by='date')[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].sum()
        ax[i].plot(x, df_vis['DENV_1'], color='black', label='DENV 1')
        ax[i].plot(x, df_vis['DENV_2'], color='red', label='DENV 2')
        ax[i].plot(x, df_vis['DENV_3'], color='green', label='DENV 3')
        ax[i].plot(x, df_vis['DENV_4'], color='blue', label='DENV 4')
        ax[i].set_ylabel('Monthly DENV cases')
    ax[2].legend(loc=1)
    ### Axis decorations
    ax[0].set_ylabel('Monthly DENV incidence')
    ax[0].set_title(f'{UF}')
    ax[0].set_xlim([min(x), max(x)])
    mx = max([np.nanmax(df_vis['DENV_1'].values), np.nanmax( df_vis['DENV_2'].values), np.nanmax( df_vis['DENV_3'].values), np.nanmax( df_vis['DENV_4'].values)])
    ax[2].set_ylim([0, 0.15*mx]) if not np.isnan(mx) else ax[2].set_ylim([0, 100])
    plt.savefig(f'../interim/datasus_DENV-linelist/figs/{UF}.png', dpi=300)
    plt.close()