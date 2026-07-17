import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

cluster_id = 11


###############
## Load data ##
###############

# load the case data and bin into five year age groups
breaks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
labels = ["[00-05(", "[05-10(", "[10-15(", "[15-20(", "[20-25(", "[25-30(", "[30-35(", "[35-40(", "[40-45(", "[45-50(", "[50-55(", "[55-60(", "[60-65(", "[65-70(", "[70-75(", "[75-80(", "[80-85(", "[85-90(", "[90-95(", "[95-100("]

cases = (
    pl.scan_parquet("../../data/interim/datasus_DENV-linelist/master/DENV-*")
    # groupby-sum out diagnosis/outcome
    .group_by(["date", "CD_MUN", "age"])
    .agg(pl.col("DENV_total").sum().alias("DENV_total"))
    .sort(["date", "CD_MUN", "age"])
    # bin it in age groups
    .with_columns(pl.col("age")
        .cut(breaks=breaks, labels=labels, left_closed=True)
        .alias("age_group")
        )
    .group_by(["date", "CD_MUN", "age_group"])
    .agg(pl.col("DENV_total").sum().alias("DENV_total"))
    .sort(["date", "CD_MUN", "age_group"])
    .collect(engine="streaming")
).to_pandas()

# load the age-municipality year demographic data
demo = pd.read_csv(os.path.join(abs_dir, '../../data/interim/IBGE_population/municipality-age_population_2000-2022.csv'))

# load clusters
clusters = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/large_clusters/clusters/clusters_rgint.csv'))
region = clusters.columns.to_list()[0]


#################
## Conversions ##
#################

# convert clusters to municipalities (really need to omit this necessity)
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))
mapping = mapping.merge(clusters[[region, 'cluster']], on=region, how='left')

# aggregate the age-municipality year demographic data spatially to the clusters
demo = demo.merge(mapping[["CD_MUN", "cluster"]], on="CD_MUN", how="left")
age_cols = [col for col in demo.columns if col.startswith('[')]
demo = demo.groupby(['cluster', 'year'])[age_cols].sum().reset_index()

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

# add demographics and compute incidence
cases['year'] = cases['season'].str[:4].astype(int)
demo_long = demo.melt(id_vars=['cluster', 'year'], value_vars=age_cols, var_name='age_group', value_name='population')
cases = cases.merge(demo_long, on=['cluster', 'year', 'age_group'], how='left')
cases = cases.drop(columns=['year'])
cases['DENV_incidence'] = cases['DENV_total'] / cases['population'] * 100000
cases = cases.dropna()

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

# compute median age of the population
pop_mean_age = (
    cases.groupby(['season', 'cluster'])
      .apply(lambda g: (g['age_mid'] * g['population']).sum() / g['population'].sum())
      .reset_index(name='pop_mean_age')
)

# compute mean age of infection
cases_mean = (
    cases.groupby(['season', 'cluster'])
      .apply(lambda g: np.average(g['age_mid'], weights=g['DENV_total']) 
             if g['DENV_total'].sum() > 0 else np.nan)
      .reset_index(name='cases_mean_age')
)

# compute median age of infection
def grouped_weighted_median_interp(g, weight_col):
    g = g.sort_values('age_low')
    total = g[weight_col].sum()
    if total == 0:
        return np.nan
    cumsum = g[weight_col].cumsum()
    cutoff = total / 2
    idx = cumsum.searchsorted(cutoff)
    row = g.iloc[idx]
    prev_cum = 0 if idx == 0 else cumsum.iloc[idx - 1]
    bin_count = row[weight_col]
    if bin_count == 0:
        return row['age_mid']
    # linear interpolation within bin
    frac = (cutoff - prev_cum) / bin_count
    return row['age_low'] + frac * (row['age_high'] - row['age_low'])

cases_median = (
    cases.groupby(['season', 'cluster'])
      .apply(grouped_weighted_median_interp, weight_col='DENV_total')
      .reset_index(name='cases_median_age')
)

# compute median age of population
pop_median_age = (
    cases.groupby(['season', 'cluster'])
      .apply(grouped_weighted_median_interp, weight_col='population')
      .reset_index(name='pop_median_age')
)

# merge dataframes of totals, means and medians
statistics = cases_mean.merge(
    cases_median,
    on=['season', 'cluster'],
    how='inner'
)
statistics = statistics.merge(
    totals,
    on=['season', 'cluster'],
    how='inner'
)
statistics = statistics.merge(
    pop_mean_age,
    on=['season', 'cluster'],
    how='left'
)
statistics = statistics.merge(
    pop_median_age,
    on=['season', 'cluster'],
    how='left'
)

# throw out unnecessary columns in the cases file
cases = cases[['season', 'cluster', 'age_group', 'DENV_total', 'DENV_incidence']]


####################
## Visualisations ##
####################

# visualise the evolution of the medians over time
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
    df_c['cases_median_age'],
    linestyle='-',
    color='black'
)
plt.scatter(
    df_c['season_start'],
    df_c['cases_median_age'],
    s=sizes,
    color='black'
)
plt.plot(
    df_c['season_start'],
    df_c['pop_median_age'],
    linestyle='-',
    color='red'
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
        0.75, 0.95,                      # position (in axes coords)
        f"Median age of infection: {statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['cases_median_age'].values[0]:.1f}\nTotal number of cases: {statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['total_cases'].values[0]:.0f}",
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
    ax.bar(tmp['age_group'], tmp['DENV_incidence'], alpha=1, color='black')

    ax.set_title(f'Season {season}')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim([-5,2600])
    ax.set_ylabel('Incidence per 100K')

plt.tight_layout()
plt.savefig('age_distribution.png', dpi=200)
plt.close()