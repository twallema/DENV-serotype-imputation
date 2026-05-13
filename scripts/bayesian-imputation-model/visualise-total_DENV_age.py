import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

# select cluster used in the visualisations
cluster_id = 11

####################################
## Load and preprocess input data ##
####################################

# load age-municipality-month dataset
cases = pd.read_csv(os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/mun/DENV_total_age_1999-2025_monthly_mun.csv'))

# load the age-municipality year demographic data
demo = pd.read_csv(os.path.join(abs_dir, '../../data/interim/population/municipality-age_population_2000-2022.csv'))

# load clusters
clusters = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/large_clusters/clusters/clusters_rgint.csv'))
region = clusters.columns.to_list()[0]

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

#################################
## Compute relevant quantities ##
#################################

# compute cumulative total per age group, cluster and season
cases = (
    cases.groupby(['season', 'cluster', 'age_group'], observed=True)['DENV_total']
      .sum()
      .reset_index()
)

# add demographics and compute incidence per 100K in every season, cluster and age group
cases['year'] = cases['season'].str[:4].astype(int)
demo_long = demo.melt(id_vars=['cluster', 'year'], value_vars=age_cols, var_name='age_group', value_name='population')
cases = cases.merge(demo_long, on=['cluster', 'year', 'age_group'], how='left')
cases = cases.drop(columns=['year'])
cases['DENV_incidence'] = cases['DENV_total'] / cases['population'] * 100000
cases = cases.dropna()

# compute incidence per 100K in every season, cluster and age group RELATIVE to that season/cluster's total incidence
cases["cluster_season_cases"] = (
    cases.groupby(["season", "cluster"])["DENV_total"]
    .transform("sum")
)
cases["cluster_season_population"] = (
    cases.groupby(["season", "cluster"])["population"]
    .transform("sum")
)
cases["total_incidence_100k"] = (
    cases["cluster_season_cases"] /
    cases["cluster_season_population"] * 100_000
)
cases["relative_incidence"] = (
    cases["DENV_incidence"] /
    cases["total_incidence_100k"]
)
cases = cases.drop(columns=['cluster_season_cases', 'cluster_season_population'])

# compute the average relative incidence per 100K in every cluster and age group ACROSS the seasons
mean_relative = (
    cases
    .groupby(["cluster", "age_group"], as_index=False)
    .agg(
        mean_relative_incidence=("relative_incidence", "mean")
    )
)
cases = cases.merge(
    mean_relative,
    on=["cluster", "age_group"],
    how="left"
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

# compute the total number of cases
totals = (
    cases.groupby(['season', 'cluster'], observed=True)['DENV_total']
         .sum()
         .reset_index(name='total_cases')
)

# merge dataframes of totals, means and medians
statistics = cases_mean.merge(
    cases_median,
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
statistics = statistics.merge(
    totals,
    on=['season', 'cluster'],
    how='inner'
)

#########################
## Make visualisations ##
#########################

# Relative incidence anomoly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# Keep age ordering
age_order = cases["age_group"].unique()

# Compute anomaly
cases["relative_anomaly"] = (
    cases["relative_incidence"] -
    cases["mean_relative_incidence"]
)

# Filter cluster
cluster_df = cases[cases["cluster"] == cluster_id]

# Seasons
seasons = sorted(cluster_df["season"].unique())

# Get one total incidence value per season
season_totals = (
    cluster_df
    .groupby("season", as_index=False)
    .agg(
        total_incidence_100k=("total_incidence_100k", "first")
    )
)

# Scale dot sizes
min_size = 25
max_size = 500

incidence_vals = season_totals["total_incidence_100k"]

scaled_sizes = (
    min_size +
    (incidence_vals - incidence_vals.min()) /
    (incidence_vals.max() - incidence_vals.min())
    * (max_size - min_size)
)

season_totals["dot_size"] = scaled_sizes.values

# Layout
ncols = 3
nrows = int(np.ceil(len(seasons) / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(8.3, 11.7/4*nrows),
    sharey=True,
    sharex=True
)

axes = axes.flatten()

for i, (ax, season) in enumerate(zip(axes, seasons)):

    # Get median age of infection
    median_age_infection = statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['cases_median_age'].values[0]
    median_age_pop = statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['pop_median_age'].values[0]

    # One season
    plot_df = (
        cluster_df[cluster_df["season"] == season]
        .copy()
    )

    # Preserve order
    plot_df["age_group"] = pd.Categorical(
        plot_df["age_group"],
        categories=age_order,
        ordered=True
    )

    plot_df = plot_df.sort_values("age_group")

    x = np.arange(len(plot_df))
    y = plot_df["relative_anomaly"].values

    # Positive green, negative red
    colors = np.where(y >= 0, "green", "red")

    # Bars
    ax.bar(
        x,
        y,
        color=colors,
        alpha=0.75
    )

    # Zero line
    ax.axhline(
        0,
        color="black",
        linewidth=1.5
    )

    # ---------------------------------
    # Add incidence dot in top-right
    # ---------------------------------
    dot_info = season_totals[
        season_totals["season"] == season
    ].iloc[0]

    ax.scatter(
        0.92,                     # x-position in axes coords
        0.88,                     # y-position in axes coords
        s=dot_info["dot_size"],
        color="orange",
        transform=ax.transAxes,
        zorder=5,
        edgecolor='black'
    )

    ax.text(
    0.05, 0.95,                      # position (in axes coords)
    f"Median age infections: {float(median_age_infection):.1f}\n$\Delta$ (infections vs. pop): {float(median_age_infection) - float(median_age_pop):.1f}",
    transform=ax.transAxes,          # <-- important
    fontsize=7,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=1
    )
)

    ax.set_title(f"{season}")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["age_group"], rotation=90, fontsize=7)
    if (i % ncols)==0:
        ax.set_ylabel("Rel. incidence anomoly")
    ax.set_ylim([-1,1])

# Remove unused axes
for i in range(len(seasons), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('incidence_anomoly.png', dpi=300)
plt.show()
plt.close()


# Median age of infection over time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
plt.savefig('median_evolution.png', dpi=300)
plt.close()