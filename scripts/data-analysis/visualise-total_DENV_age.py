import os
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

cluster_id = 11
start_month_season = 9
desired_age_groups = pd.IntervalIndex.from_tuples([(0, 5),(5, 15),(15, 25),(25, 45),(45, 65),(65, 120)], closed='left')


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
demo = pd.read_csv(os.path.join(abs_dir, '../../data/interim/population/municipality-age_population_2000-2025_datasus.csv'))

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

# change the age groups (str) into interval (object)
interval_cols = pd.IntervalIndex.from_tuples(
    demo.columns[2:]
    .str.extract(r'\[(\d+)-(\d+)\(')
    .astype(int)
    .apply(tuple, axis=1),
    closed='left'
)
demo.columns = ['cluster', 'year', *interval_cols]

# re-aggregate to the desired age groups (MUST BE AGGREGATIONS OF 5-year bins)
rebinned = demo[['cluster', 'year']].copy()
for coarse in desired_age_groups:
    cols_to_sum = [
        c for c in interval_cols
        if c.left >= coarse.left and c.right <= coarse.right
    ]
    rebinned[coarse] = demo[cols_to_sum].sum(axis=1)
demo = rebinned

# aggregate the age-municipality-month dataset to the clusters
cases = cases.merge(mapping[["CD_MUN", "cluster"]], on="CD_MUN", how="left")

# append a "season" label (September of year X -> September of year X+1)
cases['date'] = pd.to_datetime(cases['date'])
year = cases['date'].dt.year
month = cases['date'].dt.month
season_start = year.where(month >= start_month_season, year - 1)
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

# convert string age bins to Interval objects for ease-of-manipulation
cases['age_group'] = pd.IntervalIndex.from_tuples(
    cases['age_group']
    .str.extract(r'\[(\d+)-(\d+)\(')
    .astype(int)
    .apply(tuple, axis=1),
    closed='left'
)

# rebin age groups
cases['age_group'] = desired_age_groups.take(desired_age_groups.get_indexer(cases['age_group'].apply(lambda x: x.left)))
cases = (cases
    .groupby(
        ['season', 'cluster', 'age_group'],
        as_index=False
    )['DENV_total']
    .sum()
)

# add demographics and compute incidence per 100K in every season, cluster and age group
cases['year'] = cases['season'].str[:4].astype(int)
demo_long = demo.melt(id_vars=['cluster', 'year'], value_vars=demo.columns[2:], var_name='age_group', value_name='population')
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

# Compute anomaly
cases["relative_anomaly"] = (
    cases["relative_incidence"] -
    cases["mean_relative_incidence"]
)

# compute midpoint of age brackets
cases['age_low'] = cases['age_group'].apply(lambda x: x.left)
cases['age_mid'] = cases['age_group'].apply(lambda x: (x.left + x.right) / 2)
cases['age_high'] = cases['age_group'].apply(lambda x: x.right)

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

# throw out unnecessary columns in the cases file
cases = cases[['season', 'cluster', 'age_group', 'DENV_total', 'DENV_incidence']]


#########################
## Make visualisations ##
#########################

# Timeseries of relative incidence anomaly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Select cluster ---
selected = desired_age_groups[:4]
df_cluster = cases[cases["cluster"] == cluster_id].copy()

# --- Sort seasons (keeps time order correct if strings behave oddly) ---
df_cluster = df_cluster.sort_values(["season", "age_group"])

# --- Plot ---
plt.figure(figsize=(11.7,8.3/1.5))

sns.lineplot(
    data=df_cluster,
    x="season",
    y="relative_anomaly",
    linewidth=1,
    legend=False,
    units='age_group',
    estimator=None,
    color='grey'
)

sns.lineplot(
    data=df_cluster[df_cluster["age_group"].isin(selected)],
    x="season",
    y="relative_anomaly",
    hue="age_group",
    linewidth=3
)


# reference line
plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.title(f"Relative Incidence Anomaly by Age Group — Cluster {cluster_id}")
plt.xlabel("Season")
plt.ylabel("Relative incidence anomaly")
plt.ylim([-0.8, 0.8])
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('incidence_anomaly_timeseries.png', dpi=300)
plt.close()


# Relative incidence anomaly across age groups
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Select cluster ---
df_cluster = cases[cases["cluster"] == cluster_id].copy()

# --- Select seasons to highlight ---
selected_seasons = [
    "2007-2008",
    "2010-2011",
    "2020-2021",
    "2021-2022",
    "2022-2023"
]

# --- Sort for proper line drawing ---
df_cluster = df_cluster.sort_values(["season", "age_mid"])

# --- Plot ---
plt.figure(figsize=(11.7, 8.3/1.5))

# all seasons in grey
sns.lineplot(
    data=df_cluster,
    x="age_mid",
    y="relative_anomaly",
    units="season",
    estimator=None,
    color="grey",
    linewidth=1,
    legend=False
)

# highlighted seasons
sns.lineplot(
    data=df_cluster[df_cluster["season"].isin(selected_seasons)],
    x="age_mid",
    y="relative_anomaly",
    hue="season",
    linewidth=3
)

# reference line
plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.title(f"Relative Incidence Anomaly across Age — Cluster {cluster_id}")
plt.xlabel("Age")
plt.ylabel("Relative incidence anomaly")

# optional: use actual age-bin centers as ticks
plt.xticks(df_cluster["age_mid"].unique())

plt.tight_layout()
plt.savefig("incidence_anomaly_by_age.png", dpi=300)
plt.close()


# Relative incidence anomaly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# Keep age ordering
age_order = cases["age_group"].unique()

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
        ax.set_ylabel("Rel. incidence anomaly")
    ax.set_ylim([-1,1])

# Remove unused axes
for i in range(len(seasons), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('incidence_anomaly.png', dpi=300)
#plt.show()
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