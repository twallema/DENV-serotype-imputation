# Cornell-hierarchDENV

Here we list a description of all datasets, raw datasets are unaltered original datasets, while interim datasets are obtained by converting raw datasets using the scripts in the conversion folder.

## Raw

+ `indexP_monthlyclimate_allmuni.csv`: Index-P at the municipality level, seasonal average (monthly). Obtained from Dr. Laura Alexander. Papers supporting a correlation between index P and DENV transmission: https://pmc.ncbi.nlm.nih.gov/articles/PMC9610358/

### Skinner et al. 2023

+ `full_dataset.csv`: Dataset used in Skinner et al. (2023) Human footprint is associated with shifts in the assemblages of major vector-borne diseases. Nature Sustainability. Contains the human footprint for all except one Brazilian municipality (Lucena; 2508604) from 2013-2019. Downloaded from https://github.com/ckglidden/human-footprint-index-VBD/blob/main/data/full_dataset.csv.

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

### Human footprint

+ `human-footprint_mun/rgi/rgint.csv`: Human footprint (averaged 2013-2019) per municipality, immediate region or intermediate region.

### indexP

+ `indexP_mun.csv`: Index-P at the municipality level, seasonal average (monthly). Essentially a copy of the original data in the raw folder.

+ `indexP_rgint.csv`: Index-P at the intermediate region level, seasonal average (monthly). Computed from the municipality level by averaging with demographic weighing. 

+ `indexP_rgi.csv`: Index-P at the immediate region level, seasonal average (monthly). Computed from the municipality level by averaging with demographic weighing. 

### DENV_per_100k

+ `DENV_per_100k_mun.csv`: Total dengue incidence per 100K inhabitants at the municipality level. Made using the formatted linelist data in `data/interim/datasus_DENV-linelist` and `data/conversion/build_dengue-incidence-100k.py`.

### Nearest hypermetro

+ `nearest-hypermetro_mun.csv`: Brazilian municipalities clustered to their nearest hypermetropolitan area. Made using `data/conversion/build_closest-hypermetro-area.R` using data downloaded using the geobr package.

### Datasus DENV linelist dataset

#### UF

+ `DENV-serotypes_1996-2025_weekly/monthly_uf.csv`: Weekly or monthly total confirmed (not discarded) DENV cases at the federative unit level, as well as number of serotyped cases per DENV serotype. Generated using `DENV_datasus_conversion.py`. 

#### Muncipality

+ `DENV-serotypes_1996-2025_weekly/monthly_mun.csv`: Weekly or monthly total confirmed (not discarded) DENV cases at the municipality level, as well as number of serotyped cases per DENV serotype. Generated using `DENV_datasus_conversion.py`. 


### DTW-MDS-embeddings

Subfolders contain the output of performing DTW on the following timeseries and projecting them down using MDS.

#### DENV per 100 K

#### Index P

#### Serotypes

### Pipeline output

#### Run ID

##### Clusters

Contains all output of the max-p regionalization `~/scripts/clustering/find-clusters.py`.

##### bayesian-imputation-model_output

Contains all output of the Bayesian serotype imputation model `~/scripts/bayesian-imputation-model/fit-imputation-model.py`. The latent serotype distribution per municipality from 1996-2025 and per month -- the output of this pipeline -- is named `DENV-serotypes-imputed_1996-2025_monthly.parquet`.

## Conversion scripts

+ `build_human-footprint.py`: Averages the raw human footprint data over the years 2013-2019 and spatially aggregates from the municipality to the immediate region and intermediate regions by computing the demographically weighted average over constitutent municipalities. One missing municipality's human footprint was set to 25.

+ `datasus_DENV-linelist_conversion.py`: Script used to convert the (partly confidential) raw linelisted datasus DENV data (`~/data/raw/datasus_DENV-linelist/composite_dataset`) into a more pleasant interim format.

+ `build_geographic-dataset.py`: A script merging the Brazilian municipalities' geometries, population, population density and environmental characteristics.

+ `build_dengue-incidence-100k.py`: A script to convert the formatted linelist data in `data/interim/datasus_DENV-linelist` to the total dengue incidence per 100K inhabitants at the municipality/immediate/intermediate region level.

+ `build_indexP.py`: A script to aggregate the municipality level index P per month to the immediate and intermediate Brazilian regions.