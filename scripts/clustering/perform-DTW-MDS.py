
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tslearn.metrics import cdist_dtw
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# spatial aggregation: 'mun' (5570 municipalities), 'rgi' (508 immediate regions), 'rgint' (130 intermediate regions)
region_filename = 'rgi'
region = 'CD_RGI'
# number of dimensions to project the DTW matrix onto (bigger = better representation of DTW matrix BUT clustering becomes harder)
n_mds_components = 3
# sigma of gaussian filter used to smooth DENV incidence per 100K
sigma = 1
# z-score the DENV incidence per 100K (doesn't work well; just here to let you know I tried this)
z_score = False
# use all data
start_date = datetime(1900,1,1)
end_date = datetime(2100,1,1)


# --- Step 1: Prepare and smooth incidence time series ---

# get the data
denv = pd.read_csv(f'../../data/interim/DENV_per_100K/DENV_per_100k_{region_filename}.csv')
denv['date'] = pd.to_datetime(denv['date'])

# cut from startdate
denv = denv[((denv['date'] >= start_date) & (denv['date'] <= end_date))]

# define gaussian smoother
def smooth_series(x, sigma):
    values = x.fillna(0).to_numpy()
    return gaussian_filter1d(values, sigma=sigma)

# perform smoothing
denv['DENV_per_100k_smooth'] = (
    denv.groupby(f'{region}')['DENV_per_100k']
      .transform(lambda x: smooth_series(x, sigma=sigma))
)

# Z-score
if z_score:
    def zscore(x):
        return (x - x.mean()) / x.std(ddof=0)
    denv["DENV_per_100k_smooth"] = (
        denv.groupby(f'{region}')["DENV_per_100k_smooth"]
        .transform(zscore)
    )

# # visualise results
# fig,ax=plt.subplots()
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 530001]['DENV_per_100k'], color='black')
# ax.plot(denv.date.unique(), denv[denv['CD_RGI'] == 530001]['DENV_per_100k_smooth'], color='red')
# plt.show()
# plt.close()



# --- Step 2: Compute DTW distance matrix ---

# pivot to wide format
ts = denv.pivot(index=f'{region}', columns='date', values='DENV_per_100k_smooth')

# tslearn expects 3D array: (n_ts, n_timesteps, 1)
X = ts.fillna(0).to_numpy()[:, :, np.newaxis]

# compute pairwise DTW distances
dtw_dist = cdist_dtw(X, sakoe_chiba_radius=1, n_jobs=-1, verbose=True)

# visualise raw matrx
plt.figure(figsize=(10, 8))
plt.imshow(dtw_dist, cmap="viridis", aspect="auto")
plt.colorbar(label="DTW distance")
plt.title("DTW distance matrix across 508 regions")
plt.axis("off")  # hide axis labels since 508 is too dense
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/DTW-mat-raw_{region_filename}.pdf')
plt.close()

# visualise clustermap
sns.clustermap(dtw_dist, cmap="viridis", figsize=(12, 12))
plt.savefig(f'../../data/interim/DTW-MDS-embeddings/DTW-mat-clustermap_{region_filename}.pdf')
plt.close()



# --- Step 3: Multidimensional Scaling (MDS) ---

# perform MDS
mds = MDS(n_components=n_mds_components, dissimilarity="precomputed", random_state=42, max_iter=1000, normalized_stress=True)
coords = mds.fit_transform(dtw_dist)
# evaluate performance metric (0.025=excellent, 0.05=good, 0.10=fair, 0.20=poor)
print(mds.stress_)
# convert to dataframe
embedding = pd.DataFrame(coords, index=ts.index, columns=[f"mds{i+1}" for i in range(n_mds_components)]).reset_index()
# save dataframe
embedding.to_csv(f'../../data/interim/DTW-MDS-embeddings/DTW-MDS-embedding_{region_filename}.csv', index=False)