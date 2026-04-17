
"""
This scripts ...
"""

import os
import unicodedata
import numpy as np
import pandas as pd

abs_dir = os.path.dirname(__file__)

# load in the raw IBGE data
pop = pd.read_csv('../raw/IBGE_population/tabela200.csv')

# change the missing values "-" into zero
cols = pop.columns.drop("Município")
pop[cols] = pop[cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# split the state and municipality
pop[["Município", "ABBREV_UF"]] = pop["Município"].str.extract(r"^(.*)\s+\((\w{2})\)$")
pop = pop.rename(columns={'Município': 'NM_MUN'})

# load in the post-2017 municipalities
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))

# convert all names to lower case without accents
def normalize(s):
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")  # accents
    s = s.lower().strip()   # lowercase + strip + remove extra spaces
    return s
pop["NM_MUN"] = pop["NM_MUN"].apply(normalize)
mapping["NM_MUN"] = mapping["NM_MUN"].apply(normalize)

# attach the state name or code
merged = pop.merge(
    mapping[["CD_MUN", "NM_MUN", "ABBREV_UF"]],
    on=["NM_MUN", "ABBREV_UF"],
    how="left",
    indicator=True
)

# identify the unmatched municipalities in `pop`
unmatched_pop = merged[merged["_merge"] == "left_only"]
unmatched_pop_list = unmatched_pop[["NM_MUN", "ABBREV_UF"]]

# identify the unmatched municipalities in `mapping`
mapping_check = mapping.merge(
    pop[["NM_MUN", "ABBREV_UF"]],
    on=["NM_MUN", "ABBREV_UF"],
    how="left",
    indicator=True
)
unmatched_mapping = mapping_check[mapping_check["_merge"] == "left_only"]
unmatched_mapping_list = unmatched_mapping[["NM_MUN", "ABBREV_UF", "CD_MUN"]]

# diagnostics
print("Matched:", (merged["_merge"] == "both").sum())
print("Unmatched pop:", len(unmatched_pop), list(unmatched_pop['NM_MUN'].values))
print("Unmatched mapping:", len(unmatched_mapping), list(unmatched_mapping['NM_MUN'].values))

# conflict resolution
## unmatched in `pop`
### 1. Cococi: Ghost town, no longer a municipality. Incorporated in municipality Tauá. Population count in 2000 was zero inhabitants.
# print(pop[pop['NM_MUN'] == 'cococi'])
pop = merged
pop = pop[pop['NM_MUN'] != 'cococi']

## unmatched in `mapping`
### 1. Mojui dos campos
### --> Split from Santarem in 2013. Both are in the same immediate and intermediate regions.
### Populations: Santarem (361 000), Mojui dos campos (16 000)
pop.loc[len(pop)+1] = ['mojui dos campos',] + \
                        list((16/361) * np.squeeze(pop[pop['NM_MUN'] == 'santarem'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'mojui dos campos']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'mojui dos campos']['CD_MUN'].values[0], np.nan]
pop.loc[pop['NM_MUN'] == 'santarem'] = ['santarem',] + \
                        list((1-(16/361)) * np.squeeze(pop[pop['NM_MUN'] == 'santarem'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'santarem']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'santarem']['CD_MUN'].values[0], np.nan]

### 2. Pescaria Brava
### --> Split from Laguna in 2013. Both are in the same immediate and intermediate regions.
### Populations: Laguna (43 000). Pescaria Brava (10 000).
pop.loc[len(pop)+1] = ['pescaria brava',] + \
                        list((10/43) * np.squeeze(pop[pop['NM_MUN'] == 'laguna'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'pescaria brava']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'pescaria brava']['CD_MUN'].values[0], np.nan]
pop.loc[pop['NM_MUN'] == 'laguna'] = ['laguna',] + \
                        list((1-(10/43)) * np.squeeze(pop[pop['NM_MUN'] == 'laguna'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'laguna']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'laguna']['CD_MUN'].values[0], np.nan]

### 3. Balneario Rincao
### --> Split from Icara in 2013. Both are in the same immediate and intermediate regions.
### Populations: Icara (63 000). Balneario Rincao (18 000).
pop.loc[len(pop)+1] = ['balneario rincao',] + \
                        list((18/63) * np.squeeze(pop[pop['NM_MUN'] == 'icara'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'balneario rincao']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'balneario rincao']['CD_MUN'].values[0], np.nan]
pop.loc[pop['NM_MUN'] == 'icara'] = ['icara',] + \
                        list((1-(18/63)) * np.squeeze(pop[pop['NM_MUN'] == 'icara'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'icara']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'icara']['CD_MUN'].values[0], np.nan]

### 4. Pinto Bandeira
### --> Split from Bento Goncalves in 2013. Both are in the same immediate and intermediate regions.
### Populations: Bento Goncalves (123 000). Pinto Bandeira (3 000).
pop.loc[len(pop)+1] = ['pinto bandeira',] + \
                        list((3/123) * np.squeeze(pop[pop['NM_MUN'] == 'bento goncalves'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'pinto bandeira']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'pinto bandeira']['CD_MUN'].values[0], np.nan]
pop.loc[pop['NM_MUN'] == 'bento goncalves'] = ['bento goncalves',] + \
                        list((1-(3/123)) * np.squeeze(pop[pop['NM_MUN'] == 'bento goncalves'].values)[1:22]) + \
                            [mapping[mapping['NM_MUN'] == 'bento goncalves']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'bento goncalves']['CD_MUN'].values[0], np.nan]

### 5. Paraiso das Aguas
### New municipality made from territories of Agua Clara, Costa Rica and Chapadao do Sul. All belong to the same intermediate region.
### Agua Clara and Costa Rica belong to the same immediate region. Chapadao do Sul and Paraiso das Aguas belong to the same immediate region.
### The population of Paraiso das Aguas is small (5000) inhabitants.
### We set the population of Paraiso das Aguas to zero. When clustering at the intermediate regions this does not influence outcomes at all.
### When clustering at the immediate regions, if the immediate regions of Agua Clara/Cost Rica and Chapadao do Sul/Paraiso das Aguas are split across two clusters, a slight mismatch may occur.
### However, given the small population sizes involved, and the generally large clusters used in this pipeline, this should not amount to more than a rounding error.
pop.loc[len(pop)+1] = ['paraiso das aguas',] + \
                        list((16/361) * np.zeros(21)) + \
                            [mapping[mapping['NM_MUN'] == 'paraiso das aguas']['ABBREV_UF'].values[0], mapping[mapping['NM_MUN'] == 'paraiso das aguas']['CD_MUN'].values[0], np.nan]

# format muni code as int
pop['CD_MUN'] = pop['CD_MUN'].astype(int)

# sort in same order as the mapping before saving
pop = (
    pop.set_index("CD_MUN")
       .loc[mapping["CD_MUN"]]   # enforce mapping order
       .reset_index()
)
pop = pop.drop(columns=["_merge", "ABBREV_UF", "NM_MUN"])

# save the dataset
pop.to_csv(os.path.join(abs_dir, f'../../data/interim/IBGE_population/municipality-age_population_2000.csv'), index=False)