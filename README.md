# DENV-serotype-imputation

## Setup and activate the conda environment

Update conda to make sure your version is up-to-date, in a terminal window do,

    ```bash
    conda update conda
    ```

Setup/update the `environment`: All dependencies needed to run the scripts are collected in the conda `hierarchSIR_env.yml` file. To set up the environment,

    ```bash
    conda env create -f DENV-SEROTYPE-IMPUTATION.yml
    conda activate DENV-SEROTYPE-IMPUTATION
    ```

or alternatively, to update the environment (needed after adding a dependency),

    ```bash
    conda activate DENV-SEROTYPE-IMPUTATION
    conda env update -f DENV-SEROTYPE-IMPUTATION.yml --prune
    ```

## Running the pipeline

### Converting raw linelist data to interim data

1. Place the raw linelist data from the Bento lab box `DENV > linelist_to_serotypes > data > linelist_data > composite_dataset` in the folder `~/data/raw/datasus_DENV-linelist/composite_dataset`. Note these data are partly confidential and should not be shared beyond the Bento lab. Download the Brazilian municipality geodata and place it in `~/data/raw/BR_Municipios_2023` (see README in `~/data/raw/BR_Municipios_2023` for download link).

2. Run the script `~/data/conversion/datasus_DENV-linelist_conversion.py` to convert the raw linelist data in a more suited interim format (will apear in `~/data/interim/datasus_DENV-linelist`). 

3. Run the script `build_geographic-dataset.py` to compile a geodataset containing municipality geometries, their aggregations into immediate and intermediate regions, as well as other covariates such as Koppen climate, Biome, etc. Results of this script are `~/data/interim/spatial_units_mapping.csv` and `~/data/interim/geographic-dataset.parquet`.

4. Run the script `datasus_DENV-linelist_conversion.py` to build a dataset containing the total DENV incidence per 100K inhabitants (uses the output from step 3 as its input).

### Performing Dynamic Time Warping and lower dimensional embedding through Multidimensional Scaling

5. Run the script `~/scripts/clustering/perform-DTW-MDS.py` to perform Dynamic Time Warping on the DENV per 100K incidence timeseries, yielding a square DTW distance matrix denoting telling us how similar the DENV incidence timeseries are. Then, the script embeds the DTW matrix in a `n_mds_components = 3` dimensional space (dimensionality reduction) so that it can be used as a covariate in the clustering algorithm. Note that 'rgi' denote the 508 immediate regions of Brasil, while 'rgint' denotes the 130 intermediate regions of Brazil. Output appears in `~/data/interim/DTW-MDS-embeddings`.

### Clustering

6. Run the script `~/scripts/clustering/find-clusters.py` to cluster the Brazilian municipalities, immediate regions or intermediate regions using the max-p regionalisation algorithm. Output appears in `~/data/interim/clusters`.

### Bayesian serotype imputation

7. Finally, run `~/scripts/bayesian-imputation-model/fit-imputation-model.py` to run the bayesian serotype imputation model on the clusters. Output appears in `~/data/interim/bayesian-imputation-model_output`.
