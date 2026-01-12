
"""
This scripts estimates the number of births and deaths in a Brazilian municipality per year (2001-2024) under the assumption that the birth and death rates on the municipality level are the same as the state's
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load births/deaths per UF
bd = pd.read_csv('../interim/IBGE_population-projections/IBGE_births-deaths_uf.csv')

# load the population by municipality (2001-2024)
pop = pd.read_csv('../raw/sprint_2025/datasus_population_2001_2024.csv')
pop = pop.rename(columns={'geocode': 'CD_MUN'})

# compute the number of births and deaths relative to the population in every UF
bd['births_rel'] = bd['births'] / bd['population']
bd['deaths_rel'] = bd['deaths'] / bd['population']

# merge relative birth and death rates to the municipality-year population dataframe
pop['CD_UF'] = pop['CD_MUN'].apply(lambda x: int(str(x)[0:2]))
pop = pop.merge(
    bd[['CD_UF', 'year', 'births_rel', 'deaths_rel']],
    on=['CD_UF', 'year'],
    how='left'
)

# compute absolute number of births and deaths per municipality
pop['estimated_births'] = np.round(pop['population'] * pop['births_rel'], 0).astype(int)
pop['estimated_deaths'] = np.round(pop['population'] * pop['deaths_rel'], 0).astype(int)

# format output
output = pop[['CD_MUN', 'year', 'population', 'estimated_births', 'estimated_deaths']].to_csv('../interim/IBGE_population-projections/IBGE_births-deaths_mun-estimated.csv', index=False)
