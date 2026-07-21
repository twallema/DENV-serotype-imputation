import os
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

start_month_season = 9

# desired age groups
desired_age_groups_labels = ["[00-05(", "[05-10(", "[10-15(", "[15-20(", "[20-25(", "[25-35(", "[35-45(", "[45-65(", "[65-120("]
desired_age_groups_breaks = [5, 10, 15, 20, 25, 35, 45, 65]


###############
## Load data ##
###############

# load clusters
clusters = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/pipeline_output/large_clusters/clusters/clusters_rgint.csv'))
region = clusters.columns.to_list()[0]

# convert clusters to municipalities (really need to omit this necessity)
mapping = pd.read_csv(os.path.join(abs_dir, f'../../data/interim/spatial_units_mapping.csv'))
mapping = mapping.merge(clusters[[region, 'cluster']], on=region, how='left')[['CD_MUN', 'cluster']].set_index('CD_MUN').to_dict()['cluster']

# load case data
cases = (
    pl.scan_parquet("../../data/interim/datasus_DENV-linelist/master/*.parquet")
    # no inconclusive cases
    .filter(pl.col("diagnosis") != "inconclusive")
    # groupby-sum out diagnosis/outcome
    .group_by(["date", "CD_MUN", "age"])
    .agg(pl.col("DENV_total").sum().alias("DENV_total"))
    .sort(["date", "CD_MUN", "age"])
    # aggregate to clusters
    .with_columns(pl.col("CD_MUN").replace_strict(mapping).alias("cluster"))
    .group_by(["date", "cluster", "age"])
    .agg(pl.col("DENV_total").sum())
    .sort(["date", "cluster", "age"])
    .collect(engine="streaming")
)

# load the age-municipality year demographic data
demo = (
    pl.scan_parquet('../../data/interim/demographics/population_mun-age_1999-2026.parquet')
    .sort(["CD_MUN", "year", "age"])
    # aggregate to clusters
    .with_columns(pl.col("CD_MUN").replace_strict(mapping).alias("cluster"))
    .group_by(["cluster", "year", "age"])
    .agg(pl.col("population").sum())
    .sort(["cluster", "year", "age"])
    .collect(engine="streaming")
)

#cases.write_parquet("cases.interim.parquet")
#demo.write_parquet("demo.interim.parquet")

#cases = pl.scan_parquet("cases.interim.parquet").collect()
#demo = pl.scan_parquet("demo.interim.parquet").collect()

# cap cases at 80 years of age
cases = (
    cases
    # Convert age 80+ to 80, extract year, and align types
    .with_columns(
        pl.col("date").dt.year().cast(pl.Int16).alias("year"),
        pl.when(pl.col("age") >= 80)
          .then(80)
          .otherwise(pl.col("age"))
          .cast(pl.Int8)
          .alias("age"),
        pl.col("cluster").cast(pl.Int64)
    )
    # Group by to aggregate cases for age 80+
    .group_by(["date", "year", "cluster", "age"])
    .agg(pl.col("DENV_total").sum())
)

# append demography to case data
cases = cases.join(
    demo.with_columns([
        pl.col("cluster").cast(pl.Int64),
        pl.col("year").cast(pl.Int16),
        pl.col("age").cast(pl.Int8)
    ]),
    on=["year", "cluster", "age"],
    how="left"
).sort(["date", "cluster", "age"])

# append a "season" label (September of year X -> September of year X+1)
cases = (
    cases
    .with_columns(
        # Determine the starting year of the season
        pl.when(pl.col("date").dt.month() >= start_month_season)
        .then(pl.col("date").dt.year())
        .otherwise(pl.col("date").dt.year() - 1)
        .alias("season_start")
    )
    .with_columns(
        # Format as "YYYY-YYYY" string label
        pl.concat_str([pl.col("season_start"), pl.col("season_start") + 1], separator="-").alias("season")
    )
    .drop("season_start")
    .filter(pl.col("season") != "2026-2027")
)

# sum cases / average population per season
cases = (
    cases.group_by(["season", "cluster", "age"])
    .agg(
        pl.col("DENV_total").sum(),
        pl.col("population").mean().alias("population")
    )
    .sort(["season", "cluster", "age"])
)

# mean age of population and cases + totals
statistics = (
    cases.group_by(["season", "cluster"])
    .agg(
        mean_infection_age=(
            (pl.col("age") * pl.col("DENV_total")).sum()
            / pl.col("DENV_total").sum()
        ),
        mean_population_age=(
            (pl.col("age") * pl.col("population")).sum()
            / pl.col("population").sum()
        ),
        DENV_total=pl.col("DENV_total").sum()
    )
    .filter(pl.col("season") != "2026-2027")
    .sort(["season", "cluster"])
)

# bin cases into desired age groups
cases = (
    cases
    .with_columns(
        pl.col("age")
        .cut(breaks=desired_age_groups_breaks, labels=desired_age_groups_labels, left_closed=True).alias("age_group")
        )
    .group_by(["season", "cluster", "age_group"])
    .agg(pl.col("DENV_total").sum(), pl.col("population").sum())
    .sort(["season", "cluster", "age_group"])
)

# compute incidence per 100K in every cluster, season, age group
cases = cases.with_columns((pl.col("DENV_total") / pl.col("population") * 100_000).alias("DENV_incidence_100k"))

# compute incidence per 100K of every age group relative to the total incidence per 100k in that cluster + season
cases = (
    cases
    # 1. Total incidence per 100k across all age groups for each season & cluster
    .with_columns(DENV_incidence_100k_season_cluster = (pl.col("DENV_total").sum().over(["season", "cluster"]) / pl.col("population").sum().over(["season", "cluster"]) * 100_000))
    # 2. Ratio of age-group incidence to overall cluster-season incidence
    .with_columns(rel_incidence = pl.col("DENV_incidence_100k") / pl.col("DENV_incidence_100k_season_cluster"))
)

# compute anomaly of a season
cases = (
    cases
    # 1. Mean relative incidence across all seasons for each cluster & age_group
    .with_columns(mean_rel_incidence = pl.col("rel_incidence").fill_nan(None).mean().over(["cluster", "age_group"]))
    # 2. Compute the anomaly
    .with_columns(rel_anomaly = pl.col("rel_incidence") - pl.col("mean_rel_incidence"))
)

# convert to pandas
cases = cases.to_pandas()
statistics = statistics.to_pandas()


#########################
## Make visualisations ##
#########################

# Timeseries of relative incidence anomaly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

highlight_age_groups = ["[00-01(", "[01-05(", "[05-10(", "[10-15(", "[15-20(", "[20-25("]

for cluster_id in cases['cluster'].unique():

    # --- Select cluster ---
    df_cluster = cases[cases["cluster"] == cluster_id].copy()

    # --- Sort seasons (keeps time order correct if strings behave oddly) ---
    df_cluster = df_cluster.sort_values(["season", "age_group"])

    # --- Plot ---
    plt.figure(figsize=(11.7,8.3/1.5))

    sns.lineplot(
        data=df_cluster,
        x="season",
        y="rel_anomaly",
        linewidth=1,
        legend=False,
        units='age_group',
        estimator=None,
        color='grey'
    )

    sns.lineplot(
        data=df_cluster[df_cluster["age_group"].isin(highlight_age_groups)],
        x="season",
        y="rel_anomaly",
        hue="age_group",
        linewidth=3
    )


    # reference line
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.title(f"Relative Incidence Anomaly by Age Group — Cluster {cluster_id}")
    plt.xlabel("Season")
    plt.ylabel("Relative incidence anomaly")
    plt.ylim([-1, 1])
    plt.xticks(rotation=90)

    plt.tight_layout()
    os.makedirs("fig/incidence_anomaly_timeseries", exist_ok=True)
    plt.savefig(f'fig/incidence_anomaly_timeseries/{cluster_id}.png', dpi=300)
    plt.close()


# Relative incidence anomaly across age groups
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for cluster_id in cases['cluster'].unique():

    # --- Select cluster ---
    df_cluster = cases[cases["cluster"] == cluster_id].copy()

    # --- Select seasons to highlight ---
    highlight_seasons = [
        "2007-2008",
        "2010-2011",
        "2020-2021",
        "2021-2022",
        "2022-2023"
    ]

    # --- Sort for proper line drawing ---
    df_cluster = df_cluster.sort_values(["season", "age_group"])

    # --- Plot ---
    plt.figure(figsize=(11.7, 8.3/1.5))

    # all seasons in grey
    sns.lineplot(
        data=df_cluster,
        x="age_group",
        y="rel_anomaly",
        units="season",
        estimator=None,
        color="grey",
        linewidth=1,
        legend=False
    )

    # highlighted seasons
    sns.lineplot(
        data=df_cluster[df_cluster["season"].isin(highlight_seasons)],
        x="age_group",
        y="rel_anomaly",
        hue="season",
        linewidth=3
    )

    # reference line
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.title(f"Relative Incidence Anomaly across Age — Cluster {cluster_id}")
    plt.xlabel("Age")
    plt.ylabel("Relative incidence anomaly")
    plt.ylim([-1, 1])
    plt.tight_layout()
    os.makedirs("fig/incidence_anomaly_age", exist_ok=True)
    plt.savefig(f'fig/incidence_anomaly_age/{cluster_id}.png', dpi=300)
    plt.close()


# Relative incidence anomaly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# Keep age ordering
age_order = cases["age_group"].unique()

# Filter cluster
for cluster_id in cases['cluster'].unique():

    cluster_df = cases[cases["cluster"] == cluster_id]

    # Seasons
    seasons = sorted(cluster_df["season"].unique())

    # Get one total incidence value per season
    season_totals = (
        cluster_df
        .groupby("season", as_index=False)
        .agg(
            DENV_incidence_100k_season_cluster=("DENV_incidence_100k_season_cluster", "first")
        )
    )

    # Scale dot sizes
    min_size = 25
    max_size = 500

    incidence_vals = season_totals["DENV_incidence_100k_season_cluster"]

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
        median_age_infection = statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['mean_infection_age'].values[0]
        median_age_pop = statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['mean_population_age'].values[0]

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
        y = plot_df["rel_anomaly"].values

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
    os.makedirs("fig/incidence_anomaly", exist_ok=True)
    plt.savefig(f'fig/incidence_anomaly/{cluster_id}.png', dpi=300)
    plt.close()


# Median age of infection over time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for cluster_id in cases['cluster'].unique():

    # visualise the evolution of the medians over time
    df_c = statistics[statistics['cluster'] == cluster_id].copy()
    df_c['season_start'] = df_c['season'].str[:4].astype(int)
    df_c = df_c.sort_values('season_start')
    # handle zeros safely for log
    sizes = np.sqrt(df_c['DENV_total'])
    # scale sizes to something visually reasonable
    sizes = sizes * 2
    plt.figure(figsize=(10, 5))
    plt.plot(
        df_c['season_start'],
        df_c['mean_infection_age'],
        linestyle='-',
        color='black'
    )
    plt.scatter(
        df_c['season_start'],
        df_c['mean_infection_age'],
        s=sizes,
        color='black'
    )
    plt.plot(
        df_c['season_start'],
        df_c['mean_population_age'],
        linestyle='-',
        color='red'
    )
    plt.xlabel('Season')
    plt.ylabel('Mean age')
    plt.title(f'Mean age over seasons (cluster {cluster_id})')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("fig/mean_age_evolution", exist_ok=True)
    plt.savefig(f'fig/mean_age_evolution/{cluster_id}.png', dpi=300)
    plt.close()

# for cluster_id in cases['cluster'].unique():

#     # visualisation helper
#     seasons = cases['season'].unique()

#     fig, axes = plt.subplots(len(seasons), 1, figsize=(12, 3 * len(seasons)), sharex=True)

#     if len(seasons) == 1:
#         axes = [axes]

#     for ax, season in zip(axes, seasons):

#         ax.text(
#             0.75, 0.95,                      # position (in axes coords)
#             f"Median age of infection: {statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['mean_infection_age'].values[0]:.1f}\nTotal number of cases: {statistics[((statistics['season'] == season) & (statistics['cluster'] == cluster_id))]['DENV_total'].values[0]:.0f}",
#             transform=ax.transAxes,          # <-- important
#             fontsize=12,
#             verticalalignment='top',
#             bbox=dict(
#                 boxstyle='round',
#                 facecolor='white',
#                 alpha=1
#             )
#         )
#         tmp = cases[((cases['season'] == season) & (cases['cluster'] == cluster_id))]
#         ax.bar(tmp['age_group'], tmp['DENV_incidence_100k'], alpha=1, color='black')

#         ax.set_title(f'Season {season}')
#         ax.tick_params(axis='x', rotation=90)
#         ax.set_ylim([-5,2600])
#         ax.set_ylabel('Incidence per 100K')

#     plt.tight_layout()
#     plt.savefig('age_distribution.png', dpi=200)
#     plt.show()
#     plt.close()