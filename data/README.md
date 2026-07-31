# DENV-serotype-imputation

Here we list a description of all datasets, raw datasets are unaltered original datasets, while interim datasets are obtained by converting raw datasets using the scripts in the conversion folder.

## Raw

+ `indexP_monthlyclimate_allmuni.csv`: Index-P at the municipality level, seasonal average (monthly). Obtained from Dr. Laura Alexander. Papers supporting a correlation between index P and DENV transmission: https://pmc.ncbi.nlm.nih.gov/articles/PMC9610358/

+ `subdivision-names.csv`: Abbreviations of Brazilian state names (ISO 3166-2:BR). Retrieved from the International Standardization Organization (ISO): https://www.iso.org/obp/ui/#iso:pub:PUB500001:en 

### Genetic sequence databases

+ `genbank_sequences.csv`: Metadata from the Brazilian Dengue sequences in Genbank. Downloaded from https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=Dengue%20virus,%20taxid:12637&Country_s=Brazil

+ `DENV1.tsv`/`DENV2.tsv`/etc.: Metadata from the Brazilian Dengue sequences in GISAID. Downloaded by Yining Sun on 2025-12-09.

### Skinner et al. 2023

+ `full_dataset.csv`: Dataset used in Skinner et al. (2023) Human footprint is associated with shifts in the assemblages of major vector-borne diseases. Nature Sustainability. Contains the human footprint for all except one Brazilian municipality (Lucena; 2508604) from 2013-2019. Downloaded from https://github.com/ckglidden/human-footprint-index-VBD/blob/main/data/full_dataset.csv.

### BR_Municipios_2023

Shapefiles of the Brazilian municipalities, including the area codes and names of the immediate regions (508), intermediate regions (133), federative units (27) and regions (5) of Brazil. Ommitted from Github due to file size limitations. Downloaded from IBGE: https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2023/Brasil/BR_Municipios_2023.zip 

### Sprint 2025

Downloaded using the instructions under '2 - Using FTPWeb' on https://sprint.mosqlimate.org/data/.

+ `datasus_population_2001_2024.csv`: Population data (source: SVS). Files with population by Brazilian municipality and year (2001 - 2024). Source: http://tabnet.datasus.gov.br/cgi/deftohtm.exe?ibge/cnv/popsvs2024br.def 

+ `environ_vars.csv`: Environmental characteristics of the municipalities (columns 'koppen' and 'biome').

### Demographics

+ `births_2000-2024_clean.csv`: Live births per municipality from 2000-2024. Retrieved from: http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sinasc/cnv/nvbr.def. Replaced seperator ";" with ",". Removed header and footer. Renamed column header "Municipio".

+ `deaths_2000-2024_clean.csv`: Deaths per municipality from 2000-2024. Retrieved from: http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt10br.def. Replaced seperator ";" with ",". Removed header and footer. Renamed column header "Municipio".

#### Population

Population by age, municipality and year were downloaded from: https://tabnet.datasus.gov.br/cgi/tabcgi.exe?ibge/cnv/popsvs2024br.def

### Datasus DENV linelist dataset

These data are partly confidential and can be found on the Bento lab box.

### Overland travel time matrices

+ `brazil-260729.osm.pbf`:  OpenStreetMaps database for Brazil. Downloaded from: https://download.geofabrik.de/south-america/brazil.html. This dataset is too large (2 GB) for GH.


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

+ `DENV-1999_2026-month-mun.parquet`: Monthly (indexed month-end) dengue cases ('DENV_total') and serotyped cases ('DENV_x') per Brazilian municipality, diagnosis (dengue, dengue with alarm signals, severe dengue, inconclusive), diagnosis method (lab, clinical/epidemiological, unknown) and hospitalisation (False/True). Generated using `DENV_datasus_conversion.py`.

+ `DENV-1999_2026-month-mun-age_group.parquet`: Monthly (indexed month-end) dengue cases ('DENV_total') and serotyped cases ('DENV_x') per Brazilian municipality, age group, diagnosis (dengue, dengue with alarm signals, severe dengue, inconclusive), diagnosis method (lab, clinical/epidemiological, unknown) and hospitalisation (False/True). Generated using `DENV_datasus_conversion.py`. 

#### Master

+ `DENV-XXXX-month-mun-age_group.parquet`: Monthly (indexed month-end) dengue cases ('DENV_total') and serotyped cases ('DENV_x') per Brazilian municipality, year of age, diagnosis (dengue, dengue with alarm signals, severe dengue, inconclusive), diagnosis method (lab, clinical/epidemiological, unknown) and hospitalisation (False/True). Generated using `DENV_datasus_conversion.py`. 

### Demographics

+ `births_mun_1999-2026.csv`: Made from `~/data/raw/demographics/births_2000-2024_clean.csv` using `build_births-deaths.py`.

+ `deaths_mun_1999-2026.csv`: Made from `~/data/raw/demographics/deaths_2000-2024_clean.csv` using `build_births-deaths.py`.

+ `population_mun-age_1999-2026.parquet`: Made from `~/data/raw/demographics/population/pop_age_XXXX_clean.csv` using `build_population-by-age_datasus.py`.

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

### Travel time matrices

+ `travel-time_car_mean_rgint.csv`: Mean travel time between Brazilian intermediate regions in hours.

+ `travel-time_car_sd.csv`: Standard deviation of the travel time between Brazilian intermediate regions in hours.

+ `fraction_max-attempts-reached_rgint.csv`: Fraction of Monte Carlo runs that timed out unsuccesfully after reaching the maximum number of server queries. 

## Conversion scripts

+ `build_human-footprint.py`: Averages the raw human footprint data over the years 2013-2019 and spatially aggregates from the municipality to the immediate region and intermediate regions by computing the demographically weighted average over constitutent municipalities. One missing municipality's human footprint was set to 25.

+ `datasus_DENV-linelist_conversion.py`: Script used to convert the (partly confidential) raw linelisted datasus DENV data (`~/data/raw/datasus_DENV-linelist/composite_dataset`) into a more pleasant interim format.

+ `build_geographic-dataset.py`: A script merging the Brazilian municipalities' geometries, population, population density and environmental characteristics.

+ `build_dengue-incidence-100k.py`: A script to convert the formatted linelist data in `data/interim/datasus_DENV-linelist` to the total dengue incidence per 100K inhabitants at the municipality/immediate/intermediate region level.

+ `build_indexP.py`: A script to aggregate the municipality level index P per month to the immediate and intermediate Brazilian regions.

+ `build_population-by-age_census.py`: Script used to format the 2000, 2010 and 2022 census population by 5-year age groups and by municipality, and linearily intrapolate them from 2000-2022 (`~/data/interim/population/municipality-age_population_2000-2022.csv`).

+ `construct_travel-time-matrices.R`: Script used to construct travel time origin-destination matrices for the Brazilian intermediate and immediate regions. Uses a Monte Carlo approach to connect municipalities in the origin and destination regions by populated-weighted random sampling.

+ `build-nearest-largest-sampling-effort.py`: 

+ `build_closest-hypermetro-area.R`: 