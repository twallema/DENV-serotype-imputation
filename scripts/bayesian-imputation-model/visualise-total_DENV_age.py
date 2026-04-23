import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

# load age-municipality-month dataset
cases = pd.read_csv(os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/mun/DENV_total_age_1999-2025_monthly_mun.csv'))

# load clusters
clusters = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/large_clusters/clusters/clusters_rgint.csv'))
region = clusters.columns.to_list()[0]

# convert clusters to municipalities (really need to omit this necessity)
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))
mapping = mapping.merge(clusters[[region, 'cluster']], on=region, how='left')

# aggregate the age-municipality-month dataset to the clusters
cases = cases.merge(mapping[["CD_MUN", "cluster"]], on="CD_MUN", how="left")

# append a "season" label (September of year X -> September of year X+1)
start_month = 9
cases['date'] = pd.to_datetime(cases['date'])
year = cases['date'].dt.year
month = cases['date'].dt.month
season_start = year.where(month >= start_month, year - 1)
cases['season'] = season_start.astype(str) + '-' + (season_start + 1).astype(str)

# compute cumulative total per age group, cluster and season
cases = (
    cases.groupby(['season', 'cluster', 'age_group'], observed=True)['DENV_total']
      .sum()
      .reset_index()
)

# compute cases by age group as a percentage of total cases
cases['DENV_pct'] = (
    cases['DENV_total'] /
    cases.groupby(['season', 'cluster'])['DENV_total'].transform('sum')
) * 100

# compute the total number of cases
totals = (
    cases.groupby(['season', 'cluster'], observed=True)['DENV_total']
         .sum()
         .reset_index(name='total_cases')
)

# compute midpoint of age brackets
bounds = cases['age_group'].str.extract(r'\[(\d+)-(\d+)')
cases['age_low'] = bounds[0].astype(int)
cases['age_high'] = bounds[1].astype(int)
cases['age_mid'] = (cases['age_low'] + cases['age_high']) / 2

# compute mean age of infection
mean = (
    cases.groupby(['season', 'cluster'])
      .apply(lambda g: np.average(g['age_mid'], weights=g['DENV_total']) 
             if g['DENV_total'].sum() > 0 else np.nan)
      .reset_index(name='mean_age')
)

# compute median age of infection
def grouped_weighted_median_interp(g):
    g = g.sort_values('age_low')

    total = g['DENV_total'].sum()
    if total == 0:
        return np.nan

    cumsum = g['DENV_total'].cumsum()
    cutoff = total / 2

    idx = cumsum.searchsorted(cutoff)
    row = g.iloc[idx]

    prev_cum = 0 if idx == 0 else cumsum.iloc[idx - 1]
    bin_count = row['DENV_total']

    if bin_count == 0:
        return row['age_mid']

    # linear interpolation within bin
    frac = (cutoff - prev_cum) / bin_count
    return row['age_low'] + frac * (row['age_high'] - row['age_low'])

median = (
    cases.groupby(['season', 'cluster'])
      .apply(grouped_weighted_median_interp)
      .reset_index(name='median_age')
)

# merge dataframes of totals, means and medians
statistics = mean.merge(
    median,
    on=['season', 'cluster'],
    how='inner'
)
statistics = statistics.merge(
    totals,
    on=['season', 'cluster'],
    how='inner'
)

# visualise the evolution of the medians over time
cluster_id = 1
df_c = statistics[statistics['cluster'] == cluster_id].copy()
df_c['season_start'] = df_c['season'].str[:4].astype(int)
df_c = df_c.sort_values('season_start')
# handle zeros safely for log
sizes = np.sqrt(df_c['total_cases'])
# scale sizes to something visually reasonable
sizes = sizes * 2
plt.figure(figsize=(10, 5))
plt.plot(
    df_c['season_start'],
    df_c['median_age'],
    linestyle='-',
    color='black'
)
plt.scatter(
    df_c['season_start'],
    df_c['median_age'],
    s=sizes,
    color='black'
)
plt.xlabel('Season')
plt.ylabel('Median age')
plt.title(f'Median age over seasons (cluster {cluster_id})')
plt.grid(True)
plt.tight_layout()
plt.savefig('median_evolution.png', dpi=200)
plt.close()


# visualisation helper
seasons = cases['season'].unique()

fig, axes = plt.subplots(len(seasons), 1, figsize=(12, 3 * len(seasons)), sharex=True)

if len(seasons) == 1:
    axes = [axes]

for ax, season in zip(axes, seasons):

    ax.text(
        0.05, 0.95,                      # position (in axes coords)
        f"Median age of infection: {statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['median_age'].values[0]:.1f}\nTotal number of cases: {statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['total_cases'].values[0]:.0f}",
        transform=ax.transAxes,          # <-- important
        fontsize=12,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=1
        )
    )
    tmp = cases[((cases['season'] == season) & (cases['cluster'] == cluster_id))]
    ax.bar(tmp['age_group'], tmp['DENV_pct'], alpha=1, color='black')

    ax.set_title(f'Season {season}')
    ax.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig('age_distribution.png', dpi=200)