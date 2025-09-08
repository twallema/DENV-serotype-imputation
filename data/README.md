# Cornell-hierarchDENV

Here we list a description of all datasets, raw datasets are unaltered original datasets, while interim datasets are obtained by converting raw datasets using the scripts in the conversion folder.

## Raw

### BR_Municipios_2023

Shapefiles of the Brazilian municipalities, including the area codes and names of the immediate regions (508), intermediate regions (133), federative units (27) and regions (5) of Brazil. Ommitted from Github due to file size limitations. Downloaded from IBGE: https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2023/Brasil/BR_Municipios_2023.zip 

### Sprint 2025

Downloaded using the instructions under '2 - Using FTPWeb' on https://sprint.mosqlimate.org/data/.

+ `datasus_population_2001_2024.csv`: Population data (source: SVS). Files with population by Brazilian municipality and year (2001 - 2024). Source: http://tabnet.datasus.gov.br/cgi/deftohtm.exe?ibge/cnv/popsvs2024br.def 

+ `environ_vars.csv`: Environmental characteristics of the municipalities (columns 'koppen' and 'biome').

### Datasus DENV linelist dataset

These data are partly confidential and can be found on the Bento lab box.

## Interim

+ `geographic-dataset.parquet`: Compressed (brotli compression) geographical dataset. Dataset containing geometries of Brazilian municipalities, along with variables relevant for clustering. Made using `data/conversion/build_geographic-dataset.py` from the data in `data/raw/BR_Municipios_2023`.

+ `spatial_units_mapping.csv`: Area codes and names of the municipalities, immediate regions, intermediate regions, federative units and regions. Also available in `geographic-dataset.parquet` but saved seperately to lower IO burden.

### DENV_per_100k

+ `DENV_per_100k_mun.csv`: Total dengue incidence per 100K inhabitants at the municipality level. Made using the formatted linelist data in `data/interim/datasus_DENV-linelist` and `data/conversion/build_dengue-incidence-100k.py`.

### Datasus DENV linelist dataset

#### UF

+ `DENV-serotypes_1996-2025_weekly/monthly_uf.csv`: Weekly or monthly total confirmed (not discarded) DENV cases at the federative unit level, as well as number of serotyped cases per DENV serotype. Generated using `DENV_datasus_conversion.py`. 

#### Muncipality

+ `DENV-serotypes_1996-2025_weekly/monthly_mun.csv`: Weekly or monthly total confirmed (not discarded) DENV cases at the municipality level, as well as number of serotyped cases per DENV serotype. Generated using `DENV_datasus_conversion.py`. 


### DTW-MDS-embeddings

+ `DTW-MDS-embedding_mun.csv`: Made using `~/scripts/clustering/perform-DTW-MDS.py`.

### Clusters

+ `clusters_rgi/rgint.csv`: Made using `~/scripts/clustering/find-clusters.py`.

+ `adjacency_matrix_rgi/rgint.csv`: Made using `~/scripts/clustering/find-clusters.py`.

### bayesian-imputation-model_output

This folder is not on GitHub but is automatically generated when users run the Bayesian serotype imputation model `~/scripts/bayesian-imputation-model/fit-imputation-model.py`. It will contain diagnostics of the model runs, as well as the final result (latent serotype distribution).

## Conversion scripts

+ `datasus_DENV-linelist_conversion.py`: Script used to convert the (partly confidential) raw linelisted datasus DENV data (`~/data/raw/datasus_DENV-linelist/composite_dataset`) into a more pleasant interim format.

+ `build_geographic-dataset.py`: A script merging the Brazilian municipalities' geometries, population, population density and environmental characteristics.

+ `build_dengue-incidence-100k.py`: A script to convert the formatted linelist data in `data/interim/datasus_DENV-linelist` to the total dengue incidence per 100K inhabitants at the municipality level.