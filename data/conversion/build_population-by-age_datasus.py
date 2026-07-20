
"""
This script estimates the population by year of age in every BR municipality (post-2017) between 1999-2025, using datasus population data from 2000-2025.
"""

import os
import random
import unicodedata
import numpy as np
import pandas as pd

abs_dir = os.path.dirname(__file__)

# load in the post-2017 municipalities
mapping = pd.read_csv(os.path.join(abs_dir,'../interim/spatial_units_mapping.csv'))
mapping['NM_MUN'] = mapping['NM_MUN'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8').lower())

# load in the datasets
filenames = os.listdir(os.path.join(abs_dir,'../raw/demographics/population/'))
filenames.sort()

# loop over filenames
length = []
results = []
for fn in filenames:

    # extract the year from the filename
    yr = fn[8:12]

    # get the data
    df = pd.read_csv(os.path.join(abs_dir, f'../raw/demographics/population/{fn}'))

    # rename 80+ into 80
    df = df.rename(columns={'80+': '80'})

    # split the first column into code and municipality name
    df[['CD_MUN', 'NM_MUN']] = df['Municipio'].str.split(' ', n=1, expand=True)
    df = df.drop(columns=['Municipio', 'NM_MUN', 'Total'])
    df['CD_MUN'] = df['CD_MUN'].astype(int)

    # match the 6-digit DATASUS CD_MUN with a 7-digit IBGE CD_MUN
    valid_codes = list(mapping['CD_MUN'].values)
    ## Step 1: build lookup dictionary: {six_digit: [seven_digit candidates]}
    lookup = {}
    for code in valid_codes:
        six_digit = code // 10  # drop last digit
        lookup.setdefault(six_digit, []).append(code)
    ## Step 2: map with warnings
    no_hits = 0
    non_unique_hits = 0
    mapped_codes = []
    for six_code in df["CD_MUN"]:
        candidates = lookup.get(six_code, [])
        if len(candidates) == 0:
            no_hits += 1
            print(f"WARNING: no match for {six_code} in {yr}")
            mapped_codes.append(None)  # or np.nan
        elif len(candidates) == 1:
            mapped_codes.append(candidates[0])
        else:
            non_unique_hits += 1
            choice = random.choice(candidates)
            print(f"WARNING: multiple matches for {six_code} in {yr} → {candidates}, picked {choice}")
            mapped_codes.append(choice)
    df["CD_MUN"] = mapped_codes

    # add the year
    df['year'] = yr

    # from wide format to long ormat
    age_cols = [str(i) for i in range(81)]
    df = df.melt(
        id_vars=['year', 'CD_MUN'],
        value_vars=age_cols,
        var_name='age',
        value_name='population'
    )
    df['age'] = df['age'].astype(int)

    # replace "-" with zero
    df['population'] = df['population'].replace('-', 0)
    df['population'] = df['population'].astype(int)

    # save result
    results.append(df)
    length.append(len(df['CD_MUN'].unique()))

# concatenate 
df = pd.concat(results, ignore_index=True)

# assign 2025 new municipality 510183 (Boa Esperança do Norte) back to Sorriso (5107925) and Nova Ubirata (5106182) lazily (50/50 split)
if yr == '2025':
    df.loc[((df['CD_MUN'] == 5107925) & (df['year'] == yr)), 'population'] += np.round(0.5 * df.loc[((df['CD_MUN'].isna()) & (df['year'] == yr)), 'population'].values)
    df.loc[((df['CD_MUN'] == 5106182) & (df['year'] == yr)), 'population'] += np.round(0.5 * df.loc[((df['CD_MUN'].isna()) & (df['year'] == yr)), 'population'].values)

# drop 510183 (Boa Esperança do Norte) 
df = df.dropna()

# sort dataframe
df = df.sort_values(by=['CD_MUN', 'age', 'year'])

# extrapolate results to 1999
X = np.array([2000, 2001, 2002, 2003, 2004], dtype=float)
Y = df.loc[df['year'].isin(["2000", "2001", "2002", "2003", "2004"]), ('CD_MUN', 'age', 'year', 'population')]['population'].to_numpy().reshape([5570,81,5])

x = X - X.mean()
slope = np.sum(x * Y, axis=2) / np.sum(x**2)
intercept = Y.mean(axis=2) - slope * X.mean()

prediction = np.round(np.maximum(intercept + slope * 1999, 0))

muns = np.sort(df["CD_MUN"].unique())
ages = np.sort(df["age"].unique())
df1999 = pd.DataFrame({
    "CD_MUN": np.repeat(muns, len(ages)),
    "age": np.tile(ages, len(muns)),
    "year": 1999,
    "population": prediction.ravel()
})

df = (
    pd.concat([df1999, df], ignore_index=True)
      .sort_values(["CD_MUN", "age", "year"])
      .reset_index(drop=True)
)

# set types
df['year'] = pd.to_numeric(df['year'])
df = df.astype({'CD_MUN': 'int32', 'population': 'int32', 'year': 'int16', 'age': 'int8'})

# save result
df.to_parquet(os.path.join(abs_dir, f'../../data/interim/demographics/population_mun-age_1999-2025.parquet'), index=False, compression='zstd')