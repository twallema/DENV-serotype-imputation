import io, sys, os
import math
import itertools
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from contextlib import redirect_stdout
from scipy.special import softmax
from glasbey import create_palette
from matplotlib.colors import ListedColormap
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import argparse

season_start_month = 9

# bayesian imputation model
import arviz
import pymc as pm
import pytensor.tensor as pt
from patsy import dmatrix

n_draw = 25
n_tune = 25

# parse arguments
# >>>>>>>>>>>>>>>

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("-ID", type=str, help="Identifier of the pipeline run.")
parser.add_argument("-n_cores", type=int, help="Number of available CPU cores.", default=8)
parser.add_argument("-n_maxp", type=int, help="Number of max-p clustering runs to average.", default=200)
parser.add_argument("-n_repeats", type=int, help="Number of repeated within-sample validations.", default=10)
parser.add_argument("-max_iterations_sa", type=int, help="Number of simulated annealing steps.", default=10)
parser.add_argument("-spatial_aggregation", type=str, help="Spatial aggregation clustering was performed on.")
parser.add_argument("-validation_bw", type=float, help="Bandwidth around Q1, Q2 and Q3 median serotype sampling effort to sample within-sample validation areas from.", default=0.05)
parser.add_argument("-validation_n", type=int, help="Number of within-sample validation areas to sample around each Q1, Q2, Q3 +/- bandwith.", default=2)
parser.add_argument("-visualise_imputed_data", type=bool, help="Make a plot of the imputed data in the clusters (recommend disable on cluster because runtime is several minutes).", default=False)

# assign to desired variables
args = parser.parse_args()
ID = args.ID
n_cores = args.n_cores
n_maxp = args.n_maxp
n_repeats = args.n_repeats
max_iterations_sa = args.max_iterations_sa
spatial_aggregation = args.spatial_aggregation
validation_bw = args.validation_bw
validation_n = args.validation_n
visualise_imputed_data = args.visualise_imputed_data

# pipeline output folder
abs_dir = os.path.dirname(__file__) # make sure all referenced paths are relative to the location of this file and not the terminal's pwd
output_folder = os.path.join(abs_dir, f'../../data/interim/pipeline_output/{ID}/clusters/hyperoptimisation/')
# check if output dir exists, if not, make it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# make an experimental design matrix
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

grid_covariates = ["indexP_DTW", "temperature_DTW", "humidity_DTW", "human_footprint", "denv_100k_cumulative", "biome"]

threshold_values = [27.5, 40, 55, 90] # CD_RGINT: 27.5, 40, 55, 90 results in 10, 15, 20 or 25 clusters

# Generate combinations of grid_covariates (True/False) AND thresholds
covariate_combinations = list(
    itertools.product([True, False], repeat=len(grid_covariates))
)
all_combinations = list(
    itertools.product(covariate_combinations, threshold_values)
)

rows = []
for cov_combo, thresh in all_combinations:
    row = dict(zip(grid_covariates, cov_combo))
    row["threshold"] = thresh
    rows.append(row)
design_matrix = pd.DataFrame(rows)

# Add variables that are always True
design_matrix["compactness"] = True
design_matrix["nearest_largest_sampling_effort"] = True

# Handle repeats and initialize results column
design_matrix = (
    design_matrix.loc[design_matrix.index.repeat(n_repeats)]
    .assign(repeat_id=np.tile(range(1, n_repeats + 1), len(design_matrix)))
    .reset_index(drop=True)
)

# Pre-initialise the likelihood and number of clusters
design_matrix["log_likelihood"] = np.nan
design_matrix["n_clusters"] = np.nan

print(f"\nNumber of covariate combinations: {len(covariate_combinations)}")
print(f"Number of thresholds: {len(threshold_values)}")
print(f"Number of repeated within-sample validations: {n_repeats}")
print(f"Total number of runs: {len(design_matrix)}\n")

print(f"Preparing data..\n")


# helper function
# >>>>>>>>>>>>>>>

def build_co_association_matrix(regions, clusters):
    """
    Build a co-association matrix containing 1 when BR regions belong to the same cluster and 0 if they don't

    input
    -----
    - regions: list
        - list containing unique BR region codes 

    - clusters: list
        - list containing the corresponding ID of the cluster the region belongs to
        - must have the same length as regions

    output
    ------

    - association_matrix: np.ndarray
        - 2D (n x n) association matrix
    """

    # check input length
    assert len(regions) == len(clusters), '`regions` and `clusters` must have the same length'

    # Start with an  matrix of zeros
    n = len(regions)
    association_matrix = np.zeros((n,n),dtype=int) # Maybe rename; this could be confused with the adjacency matrices being created in find-clusters that show which clusters (1-36) are next to each other

    # Loop through each pair of regions and check if they are in the same cluster. Set to 1 if two regions are in the same cluster, 0 if they are not (or the regions are the same)
    for i in range(n):
        for j in range(n):
            if i != j and clusters[i] == clusters[j]:
                association_matrix[i,j] = 1
    return association_matrix


# Load raw data
# >>>>>>>>>>>>>

# Load geodata
geography = gpd.read_parquet(os.path.join(abs_dir, "../../data/interim/geographic-dataset.parquet"))
geography = geography.to_crs('EPSG:5880')

# load state boundaries
gdf_states = geography.dissolve(by='CD_UF')

# Load case data
denv = pl.scan_parquet("../../data/interim/datasus_DENV-linelist/DENV-1999_2026-month-mun-no_diagnostics.parquet").collect().to_pandas()

# Load cases per 100K data
denv_100k = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DENV_per_100k/DENV_per_100k_{spatial_aggregation}.csv'))
denv_100k['date'] = pd.to_datetime(denv_100k['date'])

# Load human footprint 
human_footprint = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/human-footprint/human-footprint_{spatial_aggregation}.csv'))

# Load DENV per 100K DTW-MDS embedding
DTW_covariates_denv_100k = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/denv_100k/DTW-MDS-embedding_{spatial_aggregation}.csv'))

# Load indexP DTW-MDS embedding
DTW_covariates_indexP = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/indexP/DTW-MDS-embedding_{spatial_aggregation}.csv'))
region = DTW_covariates_indexP.columns.to_list()[0]

# Load temperature DTW-MDS embedding
DTW_covariates_temp = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/climate/temp_med/DTW-MDS-embedding_{spatial_aggregation}.csv'))

# Load humidity DTW-MDS embedding
DTW_covariates_humid = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/climate/humid_med/DTW-MDS-embedding_{spatial_aggregation}.csv'))

# Load serotypes DTW-MDS embedding
DTW_covariates_serotypes = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/DTW-MDS-embeddings/serotypes/{region}/DTW-MDS-embedding_{spatial_aggregation}.csv'))

# Load nearest hypermetro area
nearest_hypermetro = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/nearest-hypermetro/nearest-hypermetro_{spatial_aggregation}.csv'))

# Load nearest largest sampling effort
nearest_largest_sampling_effort = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/nearest-largest-sampling-effort/nearest-largest-sampling-effort_{spatial_aggregation}.csv'))


# Aggregate incidence and geographical dataset to the intermediate/immediate regions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if region:

    assert ((region != 'CD_RGI') | (region != 'CD_RGINT')), "'region' must be either 'CD_RGI' or 'CD_RGINT''"

    # Geography
    # >>>>>>>>>

    muncipality_region_map = geography[['CD_MUN', f'{region}']]

    # --- 1. Majority vote of biome per immediate region ---
    # Count how many municipalities per biome in each immediate region
    biome_majority = (
        geography.groupby([f'{region}', 'biome'])
        .size()
        .reset_index(name='count')
    )
    # For each immediate region, keep the biome with max count
    biome_majority = (
        biome_majority
        .sort_values([f'{region}', 'count'], ascending=[True, False])
        .drop_duplicates(f'{region}')
        .set_index(f'{region}')['biome']
    )
    # --- 2. Majority vote of Koppen climate per immediate region ---
    # Count how many municipalities per koppen in each immediate region
    koppen_majority = (
        geography.groupby([f'{region}', 'koppen'])
        .size()
        .reset_index(name='count')
    )
    # For each immediate region, keep the koppen with max count
    koppen_majority = (
        koppen_majority
        .sort_values([f'{region}', 'count'], ascending=[True, False])
        .drop_duplicates(f'{region}')
        .set_index(f'{region}')['koppen']
    )
    # --- 3. Dissolve geometries by immediate region ---
    gdf_regions = geography.dissolve(by=f'{region}', aggfunc={'POP': 'sum'})
    # --- 4. Attach the majority biome and koppen back ---
    gdf_regions['biome'] = gdf_regions.index.map(biome_majority)
    gdf_regions['koppen'] = gdf_regions.index.map(koppen_majority)
    # --- 5. Retain only relevant columns ---
    gdf_regions = gdf_regions.reset_index()
    geography = gdf_regions[[f'{region}', 'biome', 'koppen', 'POP', 'geometry']]

    # Incidence
    # >>>>>>>>>

    # Merge incidence with mapping
    denv = denv.merge(muncipality_region_map, on="CD_MUN", how="left")
    # Define custom aggregation function to treat the Nans
    def nan_to_zero_sum(series):
        if series.isna().all():
            return float("nan")
        else:
            return series.fillna(0).sum()
    # List of columns to aggregate
    denv_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]
    # Group and aggregate
    denv = (
        denv.groupby([f"{region}", "date"])[denv_cols]
        .agg(nan_to_zero_sum)
        .reset_index()
    )


# Compute threshold & leave out within-sample validation
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Compute the mimimum sum of serotyped cases across all years (will have to be changed)
denv = denv[denv['date'] < datetime(2020,1,1)]
# append a season label
before_start_month = denv['date'].dt.month < season_start_month
season_year = denv['date'].dt.year
season_year = season_year.where(~before_start_month, season_year - 1)
denv['season'] = season_year.astype(str) + '-' + (season_year + 1).astype(str)
# compute total cases per month
denv["N_typed"] = denv[["DENV_1","DENV_2","DENV_3","DENV_4"]].sum(axis=1)
# sum cases by year
yearly_sum = denv.groupby([f'{region}',"season"])['N_typed'].sum().reset_index()
# take median across years
yearly_sum_median = yearly_sum.groupby(f'{region}')["N_typed"].median() # array for clustering
yearly_sum_median.rename("N_typed_yearly_median", inplace=True)

# Make biome covariate
# >>>>>>>>>>>>>>>>>>>>

# Make dummies for the biome
biome_dummies = pd.get_dummies(geography["biome"], prefix="biome")
geography = geography.merge(
    biome_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)
# ensure biome dummies are int (0/1)
for col in biome_dummies.columns:
    geography[col] = geography[col].astype(float)


# Make Koppen climate covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Make dummies for the koppen climate
koppen_dummies = pd.get_dummies(geography["koppen"], prefix="koppen")
geography = geography.merge(
    koppen_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)
# Ensure biome dummies are int (0/1)
for col in koppen_dummies.columns:
    geography[col] = geography[col].astype(float)


# Make nearest hypermetro area covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Make dummies for the nearest hypermetro covariate
nearest_hypermetro_dummies = pd.get_dummies(nearest_hypermetro['hypermetro_id'], prefix="nearest_hypermetro")
# Merge to the geography dataframe
geography = geography.merge(
    nearest_hypermetro_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)
# Ensure nearest_hypermetro dummies are int (0/1)
for col in nearest_hypermetro_dummies.columns:
    geography[col] = geography[col].astype(float)


# Make nearest largest sampling area covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Make dummies for the nearest hypermetro covariate
nearest_largest_sampling_effort_dummies = pd.get_dummies(nearest_largest_sampling_effort['largest_sampling_effort_id'], prefix="nearest_largest_sampling_effort")
# Merge to the geography dataframe
geography = geography.merge(
    nearest_largest_sampling_effort_dummies, 
    left_index=True, 
    right_index=True, 
    how="left"
)
# Ensure nearest_hypermetro dummies are int (0/1)
for col in nearest_largest_sampling_effort_dummies.columns:
    geography[col] = geography[col].astype(float)


# Make human footprint covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# standardize
sc = StandardScaler()
geography['human_footprint'] = sc.fit_transform(human_footprint[["human_footprint"]])


# Make cumulative DENV per 100K covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# compute cumulative totals
denv_100k = denv_100k.groupby(by=f'{region}')['DENV_per_100k'].sum()
# standardize
sc = StandardScaler()
geography['denv_100k_cumulative'] = sc.fit_transform(denv_100k.values.reshape(-1,1))


# Make compactness covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>

# 1) Project to Brazil Polyconic (EPSG:5880)
geography = geography.to_crs("EPSG:5880")

# 2) Compute centroids (in metres)
# .centroid is fine after projecting; for very complex multipolygons consider representative_point()
geography["cx"] = geography.geometry.centroid.x
geography["cy"] = geography.geometry.centroid.y

# 3) Standardize and add compactness components
sc = StandardScaler()
geography[["cx","cy"]] = sc.fit_transform(geography[["cx","cy"]])

# 4) Normalize the area codes (similarity in codes reflects proximity in space)
geography[region+'_NORM'] = sc.fit_transform(geography[[region]])


# Make DENV per 100k DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates_denv_100k, 
    on = f'{region}'
)
# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates_denv_100k_names = [x for x in DTW_covariates_denv_100k.columns.to_list() if x != f'{region}']
geography[DTW_covariates_denv_100k_names] = sc.fit_transform(geography[DTW_covariates_denv_100k_names])


# Make indexP DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates_indexP, 
    on = f'{region}'
)
# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates_indexP_names = [x for x in DTW_covariates_indexP.columns.to_list() if x != f'{region}']
geography[DTW_covariates_indexP_names] = sc.fit_transform(geography[DTW_covariates_indexP_names])


# Make temp_med DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates_temp, 
    on = f'{region}'
)
# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates_temp_names = [x for x in DTW_covariates_temp.columns.to_list() if x != f'{region}']
geography[DTW_covariates_temp_names] = sc.fit_transform(geography[DTW_covariates_temp_names])


# Make humid_med DTW-MDS covariate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Merge to the geography
geography = geography.merge(
    DTW_covariates_humid, 
    on = f'{region}'
)
# Standardize DTW-MDS embedding
sc = StandardScaler()
DTW_covariates_humid_names = [x for x in DTW_covariates_humid.columns.to_list() if x != f'{region}']
geography[DTW_covariates_humid_names] = sc.fit_transform(geography[DTW_covariates_humid_names])


# Make copies before looping
# >>>>>>>>>>>>>>>>>>>>>>>>>>

yearly_sum_median_copy = yearly_sum_median.copy(deep=True)
geography_copy = geography.copy(deep=True)

for repeat_id in design_matrix['repeat_id'].unique():

    geography = geography_copy
    yearly_sum_median = yearly_sum_median_copy

    os.makedirs(os.path.join(output_folder, f'repeat_{repeat_id}'), exist_ok=True)

    # Perform the training-validation split
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Randomly select `validation_n` areas from a band of Q1, Q2, Q3 +/- BW
    quartiles = [0.25, 0.50, 0.75]
    rng = np.random.default_rng()

    yearly_sum_median = yearly_sum_median.sort_values()
    validation_labels = []
    used_states_globally = set()

    for q in quartiles:
        quartile_selections = []

        lower_q = max(0.0, q - validation_bw)
        upper_q = min(1.0, q + validation_bw)
        
        lower_bound = np.quantile(yearly_sum_median, lower_q)
        upper_bound = np.quantile(yearly_sum_median, upper_q)
        
        candidates = yearly_sum_median.loc[
            (yearly_sum_median >= lower_bound) & (yearly_sum_median <= upper_bound)
        ].index.tolist()

        rng.shuffle(candidates)

        for c in candidates:
            if len(quartile_selections) >= validation_n:
                break

            state_code = str(c)[:2]
            
            if state_code not in used_states_globally and c not in validation_labels:
                quartile_selections.append(c)
                used_states_globally.add(state_code) # Lock this state globally

        # 6. Check if we failed to meet the requested N due to data constraints
        if len(quartile_selections) < validation_n:
            print(f"Warning: Only found {len(quartile_selections)} valid regions for Q={q:.2f} "
                f"due to strict global state constraints (Requested: {validation_n}).")

        # 7. Save the selections for this quartile band
        validation_labels.extend(quartile_selections)

    yearly_sum_median.loc[validation_labels] = 0

    # convert validation labels to municipality level (because the incidence data used in the bayesian model is too)
    validation_labels_muni = muncipality_region_map[muncipality_region_map[f'{region}'].isin(validation_labels)]['CD_MUN']
    validation_labels_muni.to_csv(os.path.join(output_folder, f'repeat_{repeat_id}/validation_labels.csv'), index=False)

    # visualise where the left out areas are
    geography['validation_labels'] = geography[f'{region}'].isin(validation_labels)

    fig,ax=plt.subplots()
    gdf_states.boundary.plot(ax=ax, linewidth=0.5, color="black")
    geography.boundary.plot(ax=ax, linewidth=0.1, alpha=0.3, color="black")
    geography.loc[geography['validation_labels'] == True].plot(
        linewidth=0.2,
        hatch='/////',
        color='red',
        alpha=0.8,
        edgecolor="grey",
        legend=False,
        ax=ax
    )
    ax.set_title('Areas left out during within-sample validation', fontsize=10)
    ax.axis("off")
    plt.savefig(os.path.join(output_folder, f'repeat_{repeat_id}/validation_labels.svg'))
    plt.close()

    # merge them to the geography dataframe
    geography = geography.merge(
        yearly_sum_median, 
        on=f"{region}",
        how="left"
    )


    # Loop over the experimental design
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    for index, row in design_matrix[design_matrix['repeat_id'] == repeat_id][ [col for col in design_matrix.columns if col != 'log_likelihood'] ].iterrows():

        print("\n")
        print(f"\nWorking on repeat {repeat_id}, index: {index}\n")

        os.makedirs(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}'), exist_ok=True)
        
        covariate_names = [col for col, val in row.to_dict().items() if val is True] # Filter covariate columns that evaluate to True

        threshold = row['threshold']

        # convert to custom inclusion of dummies etc.
        covariate_names_raw = []
        for covname in covariate_names:

            if covname == 'biome':
                covariate_names_raw.extend(biome_dummies.columns.to_list())
            elif covname == 'koppen':
                covariate_names_raw.extend(koppen_dummies.columns.to_list())
            elif covname == 'nearest_hypermetro':
                covariate_names_raw.extend(nearest_hypermetro_dummies.columns.to_list())
            elif covname == 'nearest_largest_sampling_effort':
                covariate_names_raw.extend(nearest_largest_sampling_effort_dummies.columns.to_list())
            elif covname == 'human_footprint':
                covariate_names_raw.extend(['human_footprint',])    
            elif covname == 'denv_100k_cumulative':
                covariate_names_raw.extend(['denv_100k_cumulative',])    
            elif covname == 'compactness':
                covariate_names_raw.extend(['cx', 'cy']) 
            elif covname == 'denv_100k_DTW':
                covariate_names_raw.extend(DTW_covariates_denv_100k_names) 
            elif covname == 'indexP_DTW':
                covariate_names_raw.extend(DTW_covariates_indexP_names) 
            elif covname == 'temperature_DTW':
                covariate_names_raw.extend(DTW_covariates_temp_names) 
            elif covname == 'humidity_DTW':
                covariate_names_raw.extend(DTW_covariates_humid_names)     


        # Run max-p regionalization model `n` times in parallel
        # TODO: when packaging this code these 2 functions will go in the source code
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        from contextlib import redirect_stdout
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        from tqdm import tqdm

        def run_single_maxp(
            run_index, geography_df, region_col, covariate_names, threshold,
            max_iterations_sa
        ):
            """
            Worker function: rebuilds its own MaxP model & captures stdout inside process.
            Returns (run_index, labels, co_assoc_matrix, best_obj_value)
            """
            # Local import to avoid multiprocessing overhead
            from libpysal.weights import Rook
            from spopt.region import MaxPHeuristic

            # Build contiguity weights
            w = Rook.from_dataframe(geography_df, use_index=False)

            # Build model
            model = MaxPHeuristic(
                geography_df,
                w,
                attrs_name=covariate_names,
                threshold_name="N_typed_yearly_median",
                threshold=threshold,
                top_n=2,
                policy="multiple",
                max_iterations_construction=100,
                max_iterations_sa=max_iterations_sa,
                verbose=True
            )

            # Capture stdout
            f = io.StringIO()
            with redirect_stdout(f):
                model.solve()

            stdout_data = f.getvalue().splitlines()
            best_vals = []

            # Parse stdout for objective value
            for i, line in enumerate(stdout_data):
                if "best objective value:" in line.lower():
                    try:
                        best_vals.append(float(stdout_data[i+1].strip()))
                    except:
                        pass
            best_obj_value = best_vals[-1] if best_vals else np.nan

            # Build outputs
            labels = model.labels_.copy()
            coassoc = build_co_association_matrix(geography_df[region_col], labels)

            return run_index, labels, coassoc, best_obj_value


        def run_parallel_maxp(n_cores, n, geography, region, covariate_names, threshold, max_iterations_sa):

            results = []
            with ProcessPoolExecutor(max_workers=n_cores) as ex:

                future_to_index = {
                    ex.submit(
                        run_single_maxp,
                        run_index=i,
                        geography_df=geography,
                        region_col=region,
                        covariate_names=covariate_names,
                        threshold=threshold,
                        max_iterations_sa=max_iterations_sa,
                    ): i
                    for i in range(n)
                }

                # Iterate over completed futures with a progress bar
                with tqdm(total=n, desc="Running Max-P optimization") as pbar:
                    for fut in as_completed(future_to_index):
                        results.append(fut.result())
                        pbar.update(1)  # Advance progress bar by 1 as each job finishes

            # Sort by run index so output ordering is guaranteed
            results.sort(key=lambda x: x[0])

            # Extract outputs
            all_labels   = [r[1] for r in results]
            all_matrices = [r[2] for r in results]
            obj_vals     = [r[3] for r in results]

            return all_labels, all_matrices, obj_vals

        # run model in parallel
        if __name__ == '__main__':
            try:
                mp.set_start_method("fork") # default unix parallel process spawn method
            except RuntimeError:
                pass
            # run func
            labels, matrices, best_obj_vals = run_parallel_maxp(
                n_cores=n_cores,
                n=n_maxp,
                max_iterations_sa=max_iterations_sa,
                geography=geography,
                region=f"{region}",
                covariate_names=covariate_names_raw,
                threshold=threshold,
            )


        # Assign weights to every run using tuned softmax
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        T = (25/1300) * (np.mean(best_obj_vals))
        weights = softmax(-np.asarray(best_obj_vals)/T)


        # Save individual runs in geography dataframe
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        for i in range(n_maxp):
            geography[f'run_{i}'] = labels[i]


        # Average co-association matrices across runs
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # compute softmax-weighted mean co-association matrix
        prob_matrix = pd.DataFrame(0.0, index=geography[region], columns=geography[region])
        for association_matrix, weight in zip(matrices, weights):
            prob_matrix += weight * association_matrix

        # save mean co-association matrix
        prob_matrix.to_csv(os.path.join(output_folder, f"repeat_{repeat_id}/index_{index}/prob_matrix_{spatial_aggregation}.csv"))

        # compute median number of clusters
        n_clusters = math.ceil(np.mean([len(np.unique(l)) for l in labels]))

        # make a categorical color palette with n_clusters distinct colors
        glasbey_cmap = ListedColormap(create_palette(palette_size=n_clusters))


        # Recluster mean co-association matrix using hierarchical clustering
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # hierarchical clustering needs distance matrix
        distance = 1 - prob_matrix
        distance = np.clip(distance, 0, 1) # some distances may become very very small negative numbers 

        # Perform hierarchical clustering (average linkage)
        Z = linkage(squareform(distance, checks=False), method='average')

        # Choose number of clusters k
        geography['consensus_clusters_hierarchical'] = fcluster(Z, n_clusters, criterion='maxclust')

        # Save clustermap
        import seaborn as sns
        sns.clustermap(1-distance, cmap='viridis')
        plt.savefig(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/clustermap_probmatrix_{spatial_aggregation}.svg'))
        plt.close()


        # Recluster mean co-association matrix using spectral clustering
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
        geography['consensus_clusters_spectral'] = sc.fit_predict(prob_matrix)+1


        # Save and visualise the mean clustering results
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # visualise clusters on a map
        fig, ax = plt.subplots(nrows=1, ncols=2)
        # hierarchical (left)
        gdf_states.boundary.plot(ax=ax[0], linewidth=0.5, color="black")   # state boundaries
        geography.plot(
            column="consensus_clusters_hierarchical",          # color regions by cluster label
            categorical=True,
            cmap=glasbey_cmap,             # categorical colormap
            linewidth=0.2,
            edgecolor="grey",
            legend=True,
            ax=ax[0],
            legend_kwds={'fontsize': 4, 'ncol': 4, 'loc': 'lower right', 'markerscale': 0.4}
        )
        ax[0].set_title(f"Hierarchical clustering", fontsize=14)
        ax[0].axis("off")
        # spectral (right)
        gdf_states.boundary.plot(ax=ax[1], linewidth=0.5, color="black")   # state boundaries
        geography.plot(
            column="consensus_clusters_spectral",          # color regions by cluster label
            categorical=True,
            cmap=glasbey_cmap,             # categorical colormap
            linewidth=0.2,
            edgecolor="grey",
            legend=False,
            ax=ax[1],
            legend_kwds={'fontsize': 4, 'ncol': 4, 'loc': 'lower right', 'markerscale': 0.4}
        )
        ax[1].set_title(f"Spectral clustering", fontsize=14)
        ax[1].axis("off")
        fig.suptitle('Consensus clusters')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/consensus_clusters_{spatial_aggregation}.svg'))
        plt.close()

        # Save the consensus clusters (hierarchical)
        clusters = geography[[f'{region}', 'consensus_clusters_hierarchical']]
        clusters = clusters.rename(columns={'consensus_clusters_hierarchical': 'cluster'})
        clusters = pd.merge(muncipality_region_map, clusters, on='CD_RGINT')
        clusters = clusters.set_index('CD_MUN').drop(columns=['CD_RGINT']).reset_index()
        clusters.to_csv(os.path.join(output_folder, f"repeat_{repeat_id}/index_{index}/clusters.csv"), index=False)


        # Save a map of Brazil with every cluster highlighted
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # make output folder
        if not os.path.exists(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/clusters')):
            os.makedirs(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/clusters'))

        # visualise clusters on a map
        for cluster_id in geography["consensus_clusters_hierarchical"].unique():
            fig, ax = plt.subplots()
            gdf_states.boundary.plot(ax=ax, linewidth=0.5, color="black")   # state boundaries
            geography.boundary.plot(ax=ax, linewidth=0.1, color="black", alpha=0.2)          # clustered spatial unit boundaries
            gdf = geography.loc[geography["consensus_clusters_hierarchical"] == cluster_id]
            gdf.plot(ax=ax, color="#d35052",edgecolor="none") # cluster
            cluster_centroid = gdf.union_all().centroid
            #ax.text(cluster_centroid.x, cluster_centroid.y, str(cluster_id), ha="center", va="center", fontsize=12, fontweight="bold", color="black")
            ax.set_axis_off()
            plt.savefig(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/clusters/cluster_{cluster_id}.svg'))
            plt.close()


        # Build the clusters' adjacency matrix needed for the Bayesian imputation model
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Step 1: Dissolve municipalities to state-level geometries
        clusters_geography = geography.dissolve(by='consensus_clusters_hierarchical', as_index=False)
        clusters_geography = clusters_geography.reset_index(drop=True)

        # Step 2: Ensure 'cluster' column is sorted
        clusters_geography = clusters_geography.sort_values('consensus_clusters_hierarchical').reset_index(drop=True)
        cluster_list = clusters_geography['consensus_clusters_hierarchical'].tolist()

        # Step 4: Build spatial index and adjacency dictionary
        sindex = clusters_geography.sindex
        adjacency = {idx: set() for idx in cluster_list}

        for i, row in clusters_geography.iterrows():
            geom_i = row.geometry
            uf_i = row['consensus_clusters_hierarchical']
            possible_matches_index = list(sindex.intersection(geom_i.bounds))
            
            for j in possible_matches_index:
                if i == j:
                    continue
                geom_j = clusters_geography.loc[j, "geometry"]
                uf_j = clusters_geography.loc[j, 'consensus_clusters_hierarchical']
                
                # Use intersects instead of touches for robustness
                if geom_i.intersects(geom_j):
                    adjacency[uf_i].add(uf_j)
                    adjacency[uf_j].add(uf_i)  # symmetric

        # Step 5: Convert to binary adjacency matrix
        adj_matrix = pd.DataFrame(0, index=cluster_list, columns=cluster_list)

        for uf in cluster_list:
            for neighbor in adjacency[uf]:
                adj_matrix.loc[uf, neighbor] = 1

        # Save in a .csv
        adj_matrix.to_csv(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/adjacency_matrix_{spatial_aggregation}.csv'))


        # Impute the case data
        # >>>>>>>>>>>>>>>>>>>>

        # write a NaN-retaining aggregation function
        agg_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4", "DENV_total"]
        agg_exprs = []
        for c in agg_cols: 
            agg_exprs.extend([
                pl.col(c).sum().alias(c),
                pl.col(c).count().alias(f"{c}_count"),  
            ])

        # make a map from CD_MUN to clusters
        muncipality_cluster_map = clusters.set_index('CD_MUN').to_dict()['cluster']

        # get case data, omit the validation dataset and do a groupby-sum to the clusters
        cases = (
            pl.scan_parquet("../../data/interim/datasus_DENV-linelist/DENV-1999_2026-month-mun-no_diagnostics.parquet")
            # set to null if CD_MUN is in the validation dataset
            .with_columns([
                pl.when(pl.col("CD_MUN").is_in(validation_labels_muni))
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in agg_cols
            ])
            # aggregate to regions
            .with_columns(pl.col("CD_MUN").replace_strict(muncipality_cluster_map).alias(f"cluster"))
            .group_by(["date", "cluster"])
            .agg(agg_exprs)
            .with_columns([
                pl.when(pl.col(f"{c}_count") == 0)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in agg_cols
            ])
            # count serotyping effort
            .with_columns(
                DENV_serotyped_count=(
                    pl.when(pl.sum_horizontal("^DENV_[1-4]$") == 0)
                    .then(None)
                    .otherwise(pl.sum_horizontal("^DENV_[1-4]$"))
                )
            )
            .drop([f"{c}_count" for c in agg_cols])
            .sort(["date", "cluster"])
            .collect()
        ).to_pandas()


        # total number of serotyped cases
        N_typed = cases.pivot(index="date", columns="cluster", values="DENV_serotyped_count").fillna(0).to_numpy().astype(int) # (n_months, n_regions)

        # number of cases per DENV serotype
        Y_list = []
        for col in ['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']:
            Y_mat = cases.pivot(index="date", columns="cluster", values=col).to_numpy()
            Y_list.append(Y_mat)
        Y_multinomial = np.stack(Y_list, axis=2).astype(int)    # (n_months, n_regions, n_serotypes)

        # lengths
        n_months = Y_multinomial.shape[0]
        n_regions = Y_multinomial.shape[1]
        n_serotypes = Y_multinomial.shape[2]

        # precision matrix
        I = pt.eye(len(adj_matrix))
        W = pt.as_tensor_variable(adj_matrix)
        D = pt.diag(pt.sum(W, axis=1))

        # build a spline basis
        t = np.arange(n_months)
        X = np.asarray(
            dmatrix(
                f"bs(t, df={int(np.round(n_months/9))}, degree=3, include_intercept=True)",
                {"t": t},
            )
        )
        n_basis = X.shape[1]
        X_pt = pt.constant(X)

        # construct model coordinates
        coords = {
            "date": cases['date'].unique(),
            "cluster": cases["cluster"].unique(),
            "serotype": np.array([1, 2, 3, 4]),
            "spline_basis": np.arange(n_basis),
        }

        # build pymc imputation model
        with pm.Model(coords=coords) as model:

            # spatially correlated spline coefficients
            psi = pm.Beta("psi", 3, 3)
            sigma_beta = pm.HalfNormal("sigma_beta", 1)
            Q = (1 - psi) * I + psi * (D - W)
            L_Q = pt.linalg.cholesky(Q)
            L_cov = pt.linalg.solve(L_Q, I)

            beta_raw = pm.Normal("beta_raw", 0, 1, shape=(n_basis, n_serotypes - 1, n_regions))
            beta_corr = sigma_beta * pt.einsum("ij,bsj->bsi", L_cov, beta_raw)
            beta = pm.Deterministic("beta", beta_corr.dimshuffle(2, 1, 0))
            
            # build splined latent state 
            theta_log = pm.Deterministic("theta_log", pt.concatenate([pt.einsum("tb,rsb->trs", X, beta), pt.zeros((n_months,n_regions,1))], axis=2), dims=("date", "cluster", "serotype"))

            # softmax splined latent state to obtain latent serotype distribution
            p = pm.Deterministic("p", pm.math.softmax(theta_log, axis=2), dims=("date", "cluster", "serotype"))

            # overdispersion model
            ## time-independent hierarchical overdispersion (per region)
            d_region_hierarch = pm.HalfNormal("d_region_hierarch", sigma=1/3)    # --> phi ~ 1000 --> low overdispersion
            d_region = pm.HalfNormal("d_region", sigma=d_region_hierarch, dims="cluster")
            phi = pm.Deterministic("phi", pt.repeat((1.0 / pm.math.maximum(d_region, 1e-12))[None, :], n_months, axis=0), dims=("date", "cluster"))
            alpha = phi[:, :, None] * p # Broadcast phi over serotypes

            # observed subtyped incidences ---
            Y_obs = pm.DirichletMultinomial("Y_obs", a=alpha, n=N_typed, observed=Y_multinomial, dims=("date", "cluster", "serotype"))

        # NUTS
        with model:
            trace = pm.sample(n_draw, tune=n_tune, target_accept=0.8, chains=4, cores=4, init='adapt_diag', progressbar=True)

        # save traces
        variables2plot = ['sigma_beta', 'psi', 'd_region_hierarch', 'd_region']
        os.makedirs(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/imputation_model/trace'), exist_ok=True)
        for var in variables2plot:
            arviz.plot_trace(trace, var_names=[var]) 
            plt.savefig(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/imputation_model/trace/trace-{var}_typing-effort-model.pdf'))
            plt.close()

        # make a posterior predictive
        with model:
            posterior_predictive = pm.sample_posterior_predictive(trace)


        # Visualise the imputed case data
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if visualise_imputed_data==True:
                
            # loop over clusters
            dates = cases['date'].unique()
            for cluster_id in cases['cluster'].unique():

                fig = plt.figure(figsize=(8.3, 11.7/6*8))
                fig.suptitle(f"Cluster: {cluster_id}")
                gs = fig.add_gridspec(8, 2)

                # map highlighting the region
                ax = fig.add_subplot(gs[0, 1])
                gdf_states.boundary.plot(ax=ax, linewidth=0.5, color="black")
                geography.boundary.plot(ax=ax, linewidth=0.1, color="black", alpha=0.2)
                geography.make_valid()
                gdf = geography.loc[geography['consensus_clusters_hierarchical'] == cluster_id]
                gdf.plot(ax=ax, color="#d35052", edgecolor="none")
                ax.set_axis_off()

                # set up rows below to span columns
                ax = []
                ax.append(fig.add_subplot(gs[1, :]))

                for r in range(2, 8):
                    ax.append(fig.add_subplot(gs[r, :], sharex=ax[0]))

                for a in ax[:-1]:
                    plt.setp(a.get_xticklabels(), visible=False)


                ax[0].plot(dates, cases.loc[cases['cluster'] == cluster_id, 'DENV_total'], color='black')
                ax[0].set_ylabel("DENV cases (-)")

                ax[1].plot(dates, cases.loc[cases['cluster'] == cluster_id, 'DENV_serotyped_count'], color='black')
                ax[1].set_ylabel("Serotyped cases (-)")

                for s in range(1,5):
                    ax[s+1].set_ylabel(f"DENV {s} (%)")
                    # data
                    ax[s+1].plot(dates, cases.loc[cases['cluster'] == cluster_id, f'DENV_{s}'].values / cases.loc[cases['cluster'] == cluster_id, 'DENV_serotyped_count'].values * 100, marker='o', markersize=2, linewidth=1, color='black')
                    # model
                    ax[s+1].plot(dates, trace.posterior['p'].median(dim=['chain', 'draw']).sel({'serotype': s, "cluster": cluster_id}).values * 100, color='red')
                    ax[s+1].fill_between(dates,
                                            trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.025).sel({'serotype': s, 'cluster': cluster_id}).values * 100,
                                            trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.975).sel({'serotype': s, 'cluster': cluster_id}).values * 100,
                                            color='red', alpha=0.1
                                        )
                    ax[s+1].fill_between(dates,
                                            trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.25).sel({'serotype': s, 'cluster': cluster_id}).values * 100,
                                            trace.posterior['p'].quantile(dim=['chain', 'draw'], q=0.75).sel({'serotype': s, 'cluster': cluster_id}).values * 100,
                                            color='red', alpha=0.2
                                        )
                    
                ax[-1].stackplot(dates, [trace.posterior['p'].mean(dim=['chain', 'draw']).sel({'serotype': serotype, "cluster": cluster_id}).values * 100 for serotype in range(1,5)], labels=['1', '2', '3', '4'], colors=['black', 'red', 'green', 'blue'], alpha=0.9)
                ax[-1].set_ylabel(f"Serotype distribution (%)")

                plt.tight_layout()
                os.makedirs(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/imputation_model/posterior_predictive'), exist_ok=True)
                plt.savefig(os.path.join(output_folder, f'repeat_{repeat_id}/index_{index}/imputation_model/posterior_predictive/cluster_{cluster_id}.pdf'))
                plt.close()


        ##############################################
        ## Compute within-sample validation metrics ##
        ##############################################

        # df with inferred serotype probability per left out municipality
        df_p = pd.DataFrame(
            index=pd.MultiIndex.from_product([dates, validation_labels_muni], names=["date", "CD_MUN"]),
            columns=['p_1', 'p_2', 'p_3', 'p_4', 'phi']
            ).reset_index()

        muncipality_cluster_map = clusters[clusters['CD_MUN'].isin(validation_labels_muni)]

        for CD_MUN_id in df_p['CD_MUN'].unique():
            cluster_id = muncipality_cluster_map.loc[muncipality_cluster_map['CD_MUN'] == CD_MUN_id, 'cluster'].values[0]
            df_p.loc[df_p['CD_MUN'] == CD_MUN_id, ('p_1', 'p_2', 'p_3', 'p_4')] = np.squeeze(trace.posterior['p'].median(dim=['chain', 'draw']).sel({"cluster": cluster_id}).values)
            df_p.loc[df_p['CD_MUN'] == CD_MUN_id, 'phi'] = trace['posterior']['phi'].mean(dim=['chain','draw']).sel({"cluster": cluster_id}).values
        df_p[['a_1', 'a_2', 'a_3', 'a_4']] = df_p[['p_1', 'p_2', 'p_3', 'p_4']].values * df_p[['phi']].values

        # dataframe with the data per left out municipality
        cases = (
            pl.scan_parquet("../../data/interim/datasus_DENV-linelist/DENV-1999_2026-month-mun-no_diagnostics.parquet")
            # set to null if CD_MUN is in the validation dataset
            .filter(pl.col("CD_MUN").is_in(validation_labels_muni))
            # count serotyping effort
            .with_columns(
                N_typed=(
                    pl.when(pl.sum_horizontal("^DENV_[1-4]$") == 0)
                    .then(None)
                    .otherwise(pl.sum_horizontal("^DENV_[1-4]$"))
                )
            )
            .sort(["date", "CD_MUN"])
            .collect()
        ).to_pandas()

        # Only evaluate log-likelihood when data are valid
        mask = ~cases[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].isna().all(axis=1)
        cases = cases.dropna(subset=['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4'], how='all')
        cases = cases.fillna(0)
        df_p = df_p.loc[mask]

        # Compute log likelihood
        from scipy.stats import dirichlet_multinomial
        x = cases[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].values
        n = np.sum(x, axis=1)
        alpha = df_p[['a_1', 'a_2', 'a_3', 'a_4']].apply(pd.to_numeric)
        logp = dirichlet_multinomial.logpmf(x=x, alpha=alpha, n=n)

        # Save result
        design_matrix.loc[index, 'log_likelihood'] = sum(logp)
        design_matrix.loc[index, 'n_clusters'] = n_clusters
        design_matrix.to_csv(os.path.join(output_folder, 'hyperoptimisation_results.csv'), index=False)