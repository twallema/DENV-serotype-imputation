# DENV-serotype-imputation

## Setup and activate the conda environment

Update conda to make sure your version is up-to-date,

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

## Running the code

### Bayesian imputation model

To run the imputation model: `python fit-model.py -ID test` (minimal; script has multiple inputs), to visualise the results: `python visualise-fit.py -state RJ -date 2025-08-27 -ID test`.