
"""
This scripts converts the raw datasus births and deaths data
"""

import random
import numpy as np
import pandas as pd
import geopandas as gpd

for bd in ['births', 'deaths']:

    # load births and deaths
    df = pd.read_csv(f'../raw/demographics/{bd}_2000-2024_clean.csv')

    # load area code mapping
    mun2uf_map = gpd.read_parquet('../interim/geographic-dataset.parquet')[['CD_UF', 'CD_MUN']].drop_duplicates().set_index('CD_MUN')['CD_UF'].to_dict()

    # extract leading 6-digit municipality code
    df["CD_MUN"] = (
        df["Municipio"]
        .str.extract(r"^(\d{6})", expand=False)
        .astype("Int64")   # nullable integer; rows without a code become <NA>
    )

    # drop the MUNICIPIO IGNORADO
    df = df.dropna()

    # convert to integers 
    numeric_cols = [col for col in df.columns.values if col not in ['Municipio', 'Total', 'CD_MUN']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0)

    # convert to a 7-code digit
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
    print(f"[DROPPED] Fraction of 6-digit municipality code with no 7-digit matches: {fraction_no_hits:.2%} ({fraction_no_hits*len(df)})")
    # Step 4: drop nan and convert to integer
    df = df.dropna(subset='CD_MUN_7')
    df['CD_MUN'] = df['CD_MUN_7'].astype(int)

    # clean columns
    del df['CD_MUN_7']
    del df['Municipio']
    del df['Total']

    # perform a regression on first five years, extrapolate to find 1999
    X = np.array([2000, 2001, 2002, 2003, 2004], dtype=float)
    Y = df[["2000", "2001", "2002", "2003", "2004"]].to_numpy(dtype=float)

    x_mean = X.mean()
    y_mean = Y.mean(axis=1)

    slope = ((X - x_mean) * (Y - y_mean[:, None])).sum(axis=1) / (
        (X - x_mean) ** 2
    ).sum()

    intercept = y_mean - slope * x_mean

    df.insert(0, "1999", intercept + slope * 1999)

    df["1999"] = df["1999"].fillna(0)

    df["1999"] = df["1999"].round().clip(lower=0)

    # perform a regression on the last five years, extrapolate to find 2025
    X = np.array([2020, 2021, 2022, 2023, 2024], dtype=float)
    Y = df[["2020", "2021", "2022", "2023", "2024"]].to_numpy(dtype=float)

    x_mean = X.mean()
    y_mean = Y.mean(axis=1)

    slope = ((X - x_mean) * (Y - y_mean[:, None])).sum(axis=1) / (
        (X - x_mean) ** 2
    ).sum()

    intercept = y_mean - slope * x_mean

    df.insert(len(df.columns)-1, "2025", intercept + slope * 2025)

    df["2025"] = df["2025"].fillna(0)

    df["2025"] = df["2025"].round().clip(lower=0)

    # convert to long format
    df = (
        df.melt(
            id_vars="CD_MUN",
            var_name="year",
            value_name=f"{bd}"
        )
        .astype({"year": int})
        .set_index(["CD_MUN", "year"])
        .sort_index()
    )

    # save result
    df.to_csv(f'../interim/demographics/{bd}_mun_1999-2025.csv', index=True)