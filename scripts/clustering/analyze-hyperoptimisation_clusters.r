library(lme4)
library(dplyr)
library(emmeans)
library(ggplot2)
library(patchwork)
library(tidyr)
library(ggnewscale)

# Set working directory to location of this script
if(!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the data
df = read.csv(file.path(getwd(), '../../data/interim/pipeline_output/CD_RGINT/results.csv'))

# Add configurations
df$configuration <- interaction(
  df$indexP_DTW,
  df$temperature_DTW,
  df$humidity_DTW,
  df$human_footprint,
  df$denv_100k_cumulative,
  df$biome,
  df$threshold,
  drop = TRUE
)

#############################################################
## Paired bootstrap to test our confidence in the "winner" ##
#############################################################

n_boot <- 3000
top_n <- 50

# Identify observed best configuration
best_config <- df %>%
  group_by(configuration) %>%
  summarise(
    mean_ll = mean(log_likelihood),
    .groups = "drop"
  ) %>%
  slice_max(mean_ll, n = 1, with_ties = FALSE) %>%
  pull(configuration)

# Perform a paired bootstrap
set.seed(123)

repeat_ids <- unique(df$repeat_id)

boot_results <- vector("list", n_boot)

for (b in seq_len(n_boot)) {
  
  # Paired bootstrap: resample repeat IDs
  sampled_repeats <- sample(
    repeat_ids,
    size = length(repeat_ids),
    replace = TRUE
  )
  
  # Number of times each repeat was sampled
  weights <- table(sampled_repeats)
  
  boot_results[[b]] <- df %>%
    mutate(
      weight = weights[as.character(repeat_id)]
    ) %>%
    filter(!is.na(weight)) %>%
    group_by(configuration) %>%
    summarise(
      mean_ll = weighted.mean(log_likelihood, weight),
      .groups = "drop"
    ) %>%
    mutate(bootstrap = b)
}

boot_results <- bind_rows(boot_results)

boot_comparisons <- boot_results %>%
  filter(configuration == best_config) %>%
  select(
    bootstrap,
    best_ll = mean_ll
  ) %>%
  right_join(
    boot_results,
    by = "bootstrap"
  ) %>%
  mutate(
    difference = mean_ll - best_ll
  )

boot_ci <- boot_comparisons %>%
  filter(configuration != best_config) %>%
  group_by(configuration) %>%
  summarise(
    estimate = mean(difference),
    lower_95 = quantile(difference, 0.025),
    upper_95 = quantile(difference, 0.975),
    lower_50 = quantile(difference, 0.25),
    upper_50 = quantile(difference, 0.75),
    .groups = "drop"
  )

bootstrap_probability <- boot_comparisons %>%
  filter(configuration != best_config) %>%
  group_by(configuration) %>%
  summarise(
    p_beats_best = mean(difference > 0),
    .groups = "drop"
  ) %>%
  arrange(desc(p_beats_best))

plot_df <- boot_ci %>%
  left_join(
    bootstrap_probability,
    by = "configuration"
  )

plot_df <- plot_df %>%
  arrange(desc(estimate)) %>%  # least negative → most negative
  mutate(
    configuration = factor(
      configuration,
      levels = configuration
    )
  )

# only show top N candidates in plot
plot_df <- plot_df %>%
  slice_head(n = top_n)

#######################
## Visualize results ##
#######################

config_info <- df %>%
  distinct(
    configuration,
    indexP_DTW,
    temperature_DTW,
    humidity_DTW,
    human_footprint,
    denv_100k_cumulative,
    biome,
    threshold
  )

config_plot_df <- plot_df %>%
  left_join(
    config_info,
    by = "configuration"
  ) %>%
  mutate(
    # Keep exactly the same row ordering as the forest plot
    configuration = factor(
      configuration,
      levels = rev(unique(plot_df$configuration))
    )
  )


# Convert the configuration columns to long format
config_long <- config_plot_df %>%
  mutate(
    indexP_DTW = as.character(indexP_DTW),
    temperature_DTW = as.character(temperature_DTW),
    humidity_DTW = as.character(humidity_DTW),
    human_footprint = as.character(human_footprint),
    denv_100k_cumulative = as.character(denv_100k_cumulative),
    biome = as.character(biome),
    threshold = as.character(threshold)

  ) %>%
  select(
    configuration,
    indexP_DTW,
    temperature_DTW,
    humidity_DTW,
    human_footprint,
    denv_100k_cumulative,
    biome,
    threshold
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      indexP_DTW,
      temperature_DTW,
      humidity_DTW,
      human_footprint,
      denv_100k_cumulative,
      biome,
      threshold
    ),
    names_to = "variable",
    values_to = "value"
  ) %>%
  mutate(
    variable = factor(
      variable,
      levels = c(
        "indexP_DTW",
        "temperature_DTW",
        "humidity_DTW",
        "human_footprint",
        "denv_100k_cumulative",
        "biome",
        "threshold"
      ),
      labels = c(
        "Index P",
        "Temp.",
        "Humidity",
        "H. Footpr.",
        "DENV/100k",
        "Biome",
        "Threshold"
      )
    ),
    included = case_when(
      variable == "Threshold" ~ NA_character_,
      value == "True" ~ "TRUE",
      value == "False" ~ "FALSE"
    ),
    symbol = case_when(
      value == "True" ~ "✓",
      value == "False" ~ "✗",
      TRUE ~ value
    )
  )

# Design matrix 
config_plot <- ggplot(
  config_long,
  aes(
    x = variable,
    y = configuration
  )
) +
  
geom_tile(
  data = filter(
    config_long,
    variable != "Threshold"
  ),
  aes(fill = included),
  width = 0.9,
  height = 0.75,
  colour = "white"
) +
  
  scale_fill_manual(
    values = c(
      "TRUE" = "#77DD77",
      "FALSE" = "#FF6961"
    ),
    guide = "none"
  ) +

  # White tick/cross inside binary cells
  geom_text(
    data = filter(
      config_long,
      variable != "Threshold"
    ),
    aes(
      label = ifelse(included == "TRUE", "+", "-")
    ),
    colour = "white",
    fontface = "bold",
    size = 4
  ) +
  
  
  # Start a new fill scale for threshold
  ggnewscale::new_scale_fill() +
  
geom_tile(
  data = filter(
    config_long,
    variable == "Threshold"
  ),
  aes(fill = as.numeric(value)),
  width = 0.9,
  height = 0.75,
  colour = "white"
) +
  
geom_text(
  data = filter(
    config_long,
    variable == "Threshold"
  ),
  aes(label = value),
  color = "white",
  fontface = "bold",
  size = 3.5
) +

scale_fill_continuous(guide = "none") +  
  
scale_x_discrete(
  position = "bottom"
) +
  
  scale_y_discrete(
    labels = NULL,
    drop = FALSE
  ) +
  
  labs(
    x = NULL,
    y = NULL
  ) +
  
  theme_classic() +
  
  theme(
    axis.text.x = element_text(
      face = "bold",
      size = 7
    ),
    axis.ticks = element_blank(),
    axis.line = element_blank(),
    plot.margin = margin(
      5.5, 0, 5.5, 5.5
    )
  )

# Forest plot part of the plot
forest <- ggplot(
  plot_df,
  aes(
    x = estimate,
    y = reorder(configuration, estimate)
  )
) +
  
  # Zero reference
  geom_vline(
    xintercept = 0,
    linetype = "dashed"
  ) +
  
  # 95% CI: thin line with whiskers
  geom_errorbar(
    aes(
      xmin = lower_95,
      xmax = upper_95
    ),
    orientation = "y",
    height = 0.15,
    linewidth = 0.35
  ) +
  
  # IQR: thick line, no whiskers
  geom_segment(
    aes(
      x = lower_50,
      xend = upper_50,
      yend = reorder(configuration, estimate)
    ),
    linewidth = 2,
    lineend = "butt"
  ) +
  
  # Point estimate
  geom_point(
    shape = 21,
    fill = "white",
    colour = "black",
    size = 2,
    stroke = 1
  ) +
  
  # Bootstrap probability
  geom_text(
    aes(
      x = max(plot_df$upper_95) + 50,
      label = sprintf(
        "%.1f%%",
        100 * p_beats_best
      )
    ),
    hjust = 0,
    size = 2.5
  ) +
  
  scale_y_discrete(
    labels = NULL
  ) +
  
  labs(
    x = "Bootstrapped difference in mean log likelihood\n(versus observed best)",
    y = NULL
  ) +
  
  theme_classic() +
  
  coord_cartesian(
    clip = "off"
  ) +
  
  theme(
    plot.margin = margin(
      5.5, 80, 5.5, 5.5
    )
  )


# 6. Combine both
final_plot <- config_plot + forest +
  plot_layout(
    widths = c(4.5,3.8)  # change ratios
  )

ggsave("result.pdf", plot=final_plot, width = 8.3, height = 11.7, units = "in")
final_plot

######################################################
## Make "Best" configuration as a standalone object ##
######################################################

# Extract the best configuration
best_design <- df %>%
  filter(configuration == best_config) %>%
  distinct(
    configuration,
    indexP_DTW,
    denv_100k_DTW,
    koppen,
    threshold
  ) %>%
  mutate(
    indexP_DTW = as.character(indexP_DTW),
    denv_100k_DTW = as.character(denv_100k_DTW),
    koppen = as.character(koppen),
    threshold = as.character(threshold)
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      indexP_DTW,
      denv_100k_DTW,
      koppen,
      threshold
    ),
    names_to = "variable",
    values_to = "value"
  ) %>%
  mutate(
    variable = factor(
      variable,
      levels = c(
        "indexP_DTW",
        "denv_100k_DTW",
        "koppen",
        "threshold"
      ),
      labels = c(
        "indexP",
        "DENV/100k",
        "Köppen",
        "Threshold"
      )
    ),
    included = case_when(
      variable == "Threshold" ~ NA_character_,
      value == "True" ~ "TRUE",
      value == "False" ~ "FALSE"
    )
  )

# Standalone best-configuration figure
best_config_plot <- ggplot(
  best_design,
  aes(
    x = variable,
    y = 1
  )
) +
  
  # TRUE/FALSE cells
  geom_tile(
    data = filter(
      best_design,
      variable != "Threshold"
    ),
    aes(fill = included),
    width = 0.9,
    height = 0.75,
    colour = "white"
  ) +
  
  scale_fill_manual(
    values = c(
      "TRUE" = "#77DD77",
      "FALSE" = "#FF6961"
    ),
    guide = "none"
  ) +
  
  # + / - symbols
  geom_text(
    data = filter(
      best_design,
      variable != "Threshold"
    ),
    aes(
      label = ifelse(
        included == "TRUE",
        "+",
        "-"
      )
    ),
    colour = "white",
    fontface = "bold",
    size = 4
  ) +
  
  # New scale for threshold
  ggnewscale::new_scale_fill() +
  
  geom_tile(
    data = filter(
      best_design,
      variable == "Threshold"
    ),
    aes(fill = as.numeric(value)),
    width = 0.9,
    height = 0.75,
    colour = "white"
  ) +
  
  scale_fill_gradient(
    low = "grey65",
    high = "grey15",
    guide = "none"
  ) +
  
  # Threshold number
  geom_text(
    data = filter(
      best_design,
      variable == "Threshold"
    ),
    aes(label = value),
    colour = "white",
    fontface = "bold",
    size = 3.5
  ) +
  
  scale_y_continuous(
    breaks = NULL
  ) +
  
  labs(
    x = NULL,
    y = NULL
  ) +
  
  theme_classic() +
  
  theme(
    axis.text.x = element_text(
      face = "bold",
      size = 9
    ),
    axis.ticks = element_blank(),
    axis.line = element_blank(),
    plot.margin = margin(
      5.5, 5.5, 5.5, 5.5
    )
  )

ggsave("best_config.pdf", plot=best_config_plot, width = 3.0, height = 11.7/22, units = "in")

#################################################################
## Try to disentangle each covariates effect with an lme model ##
#################################################################

# factor covariates
df$indexP_DTW <- factor(df$indexP_DTW, levels = c("False", "True"))
df$temperature_DTW <- factor(df$temperature_DTW, levels = c("False", "True"))
df$humidity_DTW <- factor(df$humidity_DTW, levels = c("False", "True"))
df$human_footprint <- factor(df$human_footprint, levels = c("False", "True"))
df$denv_100k_cumulative <- factor(df$denv_100k_cumulative, levels = c("False", "True"))
df$biome <- factor(df$biome, levels = c("False", "True"))

# fit linear mixed effects model
model <- lmer(
  log_likelihood ~ indexP_DTW + temperature_DTW + humidity_DTW + human_footprint + denv_100k_cumulative + biome + threshold + (1 | repeat_id),
  data = df
)

summary(model)

# compute confidence intervals
confint <- confint(model, method = "profile")

# QQ plots
qqnorm(ranef(model)$repeat_id[[1]],
       main = "QQ plot: random intercepts")
qqline(ranef(model)$repeat_id[[1]])

qqnorm(residuals(model),
       main = "QQ plot: residuals")
qqline(residuals(model))
