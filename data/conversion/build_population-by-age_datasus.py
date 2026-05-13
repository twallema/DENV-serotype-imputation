
"""
This script estimates the population by 5-year age bins in every BR municipality between 2000-2022, by interpolating the 2000, 2010 and 2022 census years.
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
filenames = os.listdir(os.path.join(abs_dir,'../raw/population/datasus/population/'))
filenames.sort()

# loop over filenames
length = []
results = []
for fn in filenames:

    # extract the year from the filename
    yr = fn[8:12]

    # get the data
    df = pd.read_csv(os.path.join(abs_dir, f'../raw/population/datasus/population/{fn}'))

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
            print(f"WARNING: no match for {six_code}")
            mapped_codes.append(None)  # or np.nan
        elif len(candidates) == 1:
            mapped_codes.append(candidates[0])
        else:
            non_unique_hits += 1
            choice = random.choice(candidates)
            print(f"WARNING: multiple matches for {six_code} → {candidates}, picked {choice}")
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

    # cut in age bins
    cutoff_age = 80
    bins = np.arange(0, cutoff_age+5, 5, dtype=int).tolist() + [120,] 
    labels = [f"[{i:02d}-{i+5:02d}(" for i in range(0, cutoff_age, 5)] + [f"[{cutoff_age}-120("]
    df['age_group'] = pd.cut(
        df['age'],
        bins=bins,
        right=False,   # intervals like [0,5)
        labels=labels,
        include_lowest=True
    )
    df_binned = (
        df
        .groupby(['year', 'CD_MUN', 'age_group'], as_index=False, observed=False)['population']
        .sum()
    )

    # save result
    results.append(df_binned)
    length.append(len(df_binned['CD_MUN'].unique()))

# concatenate 
df = pd.concat(results, ignore_index=True)

# pivot back to a wide format
df = (
    df.pivot(
        index=['CD_MUN', 'year'],
        columns='age_group',
        values='population'
    )
    .reset_index()
)

# make sure CD_MUN is int
df['CD_MUN'] = df['CD_MUN'].astype(int)

# print("Length per year:")
# print(length) # all lengths equal to 5570

df.to_csv(os.path.join(abs_dir, f'../../data/interim/population/municipality-age_population_2000-2025_datasus.csv'), index=False)