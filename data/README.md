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

+ `weighted_distance_matrix.csv`: Contains a square origin-destination-type distance matrix with the population-weighted distance between Brazil's 27 states.

+ `adjacency_matrix.csv`: Contains a square origin-destination-type adjacency matrix of the Brasilian states.

+ `geographic-dataset.parquet`: Compressed (brotli compression) geographical dataset. Dataset containing geometries of Brazilian municipalities, along with variables relevant for clustering. Made using `data/conversion/build_geographic-dataset.py` from `data/raw/BR_Municipios_2023`.

+ `spatial_units_mapping.csv`: Area codes and names of the municipalities, immediate regions, intermediate regions, federative units and regions. Also available in `geographic-dataset.parquet` but saved seperately to lower IO burden.

### Datasus DENV linelist dataset

#### UF

+ `DENV-serotypes_1996-2025_monthly/weekly_UF.csv`: Weekly or monthly total confirmed (not discarded) DENV cases at the municipality level, as well as number of serotyped cases per DENV serotype. Generated using `DENV_datasus_conversion.py`. 

#### Muncipality

+ ``: Weekly or monthly total confirmed (not discarded) DENV cases at the municipality level, as well as number of serotyped cases per DENV serotype. Generated using `DENV_datasus_conversion.py`. 


### Imputed DENV datasus

+ `DENV-serotypes-imputed_1996-2025_monthly.csv`: Monthly total confirmed DENV cases in every Brazilian UF (column: `DENV_total`), identical to `~/data/interim/DENV_datasus/DENV-serotypes_1996-2025_monthly.csv`. Contains the serotype fractions in columns `p_1`, `p_2`, `p_3` and `p_4`, as generated using the Bayesian serotype imputation model in `~/data/conversion/modeling_serotypes/fit-model.py`.

## Conversion scripts

+ `build_distance-adjacency-matrix.py`: Notebook used to build an adjacency matrix and a demographically-weighted distance matrix between Brazilian states.

+ `datasus_DENV-linelist_conversion.py`: Script used to convert the (partly confidential) raw linelisted datasus DENV data (`~/data/raw/datasus_DENV-linelist/composite_dataset`) into a more pleasant interim format.

+ `build_geographic-dataset.py`: A script merging the Brazilian municipalities' geometries, population, population density and environmental characteristics.