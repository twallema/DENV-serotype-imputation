library(sf)
library(geobr)
library(igraph)
library(ggplot2)
library(tidyverse)
library(randomcoloR)

# All paths relative to this file
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Find nearest hypermetro area for all municipalities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Download shape files of states, municipalities and metro areas
shape_states = geobr::read_state(year=2018)
shape_munis = read_municipality(year=2018)
metros <- geobr::read_metro_area(year = 2018, code_state = "all", simplified = TRUE, showProgress = TRUE, cache = TRUE) 

# Dissolve the municipalities inside metro areas
metro_polys <- metros %>%
  group_by(name_metro) %>%
  summarise(geometry = st_union(geom)) %>%
  st_as_sf()
# Filter imperfections
metro_polys <- metro_polys %>%
  mutate(geometry = st_simplify(geometry, dTolerance = 100))

# Find which metro areas share a border
touch_list <- st_intersects(metro_polys)
g <- graph_from_adj_list(touch_list)
clusters <- components(g)$membership
metro_polys$cluster_id <- clusters

# Construct hypermetro areas (33)
metro_polys <- metro_polys %>%
  group_by(cluster_id) %>%
  summarise(geometry = st_union(geometry)) %>%
  st_as_sf()
# Filter out imperfections
metro_polys <- metro_polys %>%
  mutate(geometry = st_simplify(geometry, dTolerance = 100))

# Compute closest metro area
## Reproject to distance based coordinates
metro_transform <- st_transform(metro_polys, 3857)
munis_transform <- st_transform(shape_munis, 3857)
## Find nearest metro area
nearest_idx <- st_nearest_feature(munis_transform, metro_transform)
result <- munis_transform %>%
  mutate(cluster_id = metro_transform$cluster_id[nearest_idx])

# Save result
result_no_geom <- st_drop_geometry(result)
result_no_geom <- result_no_geom %>% rename("hypermetro_id" = "cluster_id", "CD_MUN" = "code_muni")
result_no_geom <- result_no_geom[, c("CD_MUN", "hypermetro_id")]
folder <- "../interim/nearest-hypermetro"
if (!dir.exists(folder)) {
  dir.create(folder, recursive = TRUE)
}
write.csv(result_no_geom[, c("CD_MUN", "hypermetro_id")],
          file.path(folder, "nearest-hypermetro_mun.csv"),
          row.names = FALSE)
# Visualise result
## Dissolve municipalities
result <- result %>%
  group_by(cluster_id) %>%
  summarise(geom = st_union(geom)) %>%
  st_as_sf()
## Filter imperfections
result <- result %>%
  mutate(geom = st_simplify(geom, dTolerance = 100))
## Make plot
n_cols <- length(unique(result$cluster_id))
cols <- randomColor(n_cols, luminosity = "bright")
ggplot() + theme_void() + theme_bw() + 
  ggtitle("Nearest hypermetro area (municipality") +
  geom_sf(data = result, aes(fill=factor(cluster_id)), alpha=1) +
  geom_sf(data = metro_polys, color='black', alpha=0.8) +
  scale_fill_manual(values = cols) +
  theme(legend.position = "none")
## Save the plot
ggsave(file.path(folder,"nearest-hypermetro_mun.pdf"))


# Aggregate to immediate regions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Retrieve mapping available in project repository
spatial_units_mapping <- read.csv("../interim/spatial_units_mapping.csv")[, c("CD_MUN", "CD_RGI", "CD_RGINT")]
## Join hypermetro data with municipality --> RGI mapping
df_joined <- result_no_geom %>%
  left_join(spatial_units_mapping, by = "CD_MUN")
## Count hypermetro_id occurrences inside each RGI
rgi_majority <- df_joined %>%
  group_by(CD_RGI, hypermetro_id) %>%
  summarise(n = n(), .groups = "drop") %>%
## Select the hypermetro_id with the highest count per RGI
group_by(CD_RGI) %>%
  slice_max(order_by = n, n = 1, with_ties = FALSE) %>%
  ungroup()
rgi_majority <- rgi_majority[, c("CD_RGI", "hypermetro_id")] 
## Get the shapefiles of the immediate regions and merge the hypermetro ID
shape_immediate_region = read_immediate_region(year=2020)
shape_immediate_region <- shape_immediate_region %>% rename("CD_RGI" = "code_immediate")
shape_immediate_region <- shape_immediate_region %>%
  left_join(rgi_majority, by = "CD_RGI")
## Save result
write.csv(rgi_majority,
          file.path(folder, "nearest-hypermetro_rgi.csv"),
          row.names = FALSE)
## Make plot
n_cols <- length(unique(shape_immediate_region$hypermetro_id))
cols <- randomColor(n_cols, luminosity = "bright")
ggplot() + theme_void() + theme_bw() + 
  ggtitle("Nearest hypermetro area (immediate regions)") +
  geom_sf(data = shape_immediate_region, aes(fill=factor(hypermetro_id)), alpha=1) +
  geom_sf(data = metro_polys, color='black', alpha=0.8) +
  scale_fill_manual(values = cols) +
  theme(legend.position = "none")
## Save the plot
ggsave(file.path(folder,"nearest-hypermetro_rgi.pdf"))


# Aggregate to intermediate regions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Count hypermetro_id occurrences inside each RGI
rgint_majority <- df_joined %>%
  group_by(CD_RGINT, hypermetro_id) %>%
  summarise(n = n(), .groups = "drop") %>%
## Select the hypermetro_id with the highest count per RGI
group_by(CD_RGINT) %>%
  slice_max(order_by = n, n = 1, with_ties = FALSE) %>%
  ungroup()
rgint_majority <- rgint_majority[, c("CD_RGINT", "hypermetro_id")] 
## Get the shapefiles of the immediate regions and merge the hypermetro ID
shape_intermediate_region = read_intermediate_region(year=2020)
shape_intermediate_region <- shape_intermediate_region %>% rename("CD_RGINT" = "code_intermediate")
shape_intermediate_region <- shape_intermediate_region %>%
  left_join(rgint_majority, by = "CD_RGINT")
## Save result
write.csv(rgint_majority,
          file.path(folder, "nearest-hypermetro_rgint.csv"),
          row.names = FALSE)
## Make plot
n_cols <- length(unique(shape_intermediate_region$hypermetro_id))
cols <- randomColor(n_cols, luminosity = "bright")
ggplot() + theme_void() + theme_bw() + 
  ggtitle("Nearest hypermetro area (intermediate regions)") +
  geom_sf(data = shape_intermediate_region, aes(fill=factor(hypermetro_id)), alpha=1) +
  geom_sf(data = metro_polys, color='black', alpha=0.8) +
  scale_fill_manual(values = cols) +
  theme(legend.position = "none")
## Save the plot
ggsave(file.path(folder,"nearest-hypermetro_rgint.pdf"))


