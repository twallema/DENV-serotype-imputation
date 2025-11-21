library(sf)
library(geobr)
library(igraph)
library(ggplot2)
library(tidyverse)
library(randomcoloR)

# Paths relative to this file
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Download shape files of states, municipalities and metro areas
shape_states = geobr::read_state(year=2018)
shape_munis = read_municipality(year=2018)
metros <- geobr::read_metro_area(
  year = 2018,
  code_state = "all",
  simplified = TRUE,
  showProgress = TRUE,
  cache = TRUE) 

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

# Compute final result
## Distance based coordinates
metro_transform <- st_transform(metro_polys, 3857)
munis_transform <- st_transform(shape_munis, 3857)
## Use nearest feature (so not centroid distance!)
nearest_idx <- st_nearest_feature(munis_transform, metro_transform)
result <- munis_transform %>%
  mutate(cluster_id = metro_transform$cluster_id[nearest_idx])
## Save result
result_no_geom <- st_drop_geometry(result)
result_no_geom <- result_no_geom %>% rename("hypermetro_id" = "cluster_id")
folder <- "../interim/nearest-hypermetro"
if (!dir.exists(folder)) {
  dir.create(folder, recursive = TRUE)
}
write.csv(result_no_geom[, c("code_muni", "hypermetro_id")],
          file.path(folder, "nearest-hypermetro_mun.csv"),
          row.names = FALSE)


## Visualise result
### Dissolve municipalities
result <- result %>%
  group_by(cluster_id) %>%
  summarise(geom = st_union(geom)) %>%
  st_as_sf()
### Filter imperfections
result <- result %>%
  mutate(geom = st_simplify(geom, dTolerance = 100))
### Make plot
n_cols <- length(unique(result$cluster_id))
cols <- randomColor(n_cols, luminosity = "bright")
ggplot() + theme_void() + theme_bw() + 
  ggtitle("Nearest hypermetro area") +
  geom_sf(data = result, aes(fill=factor(cluster_id)), alpha=1) +
  geom_sf(data = metro_polys, color='black', alpha=0.8) +
  scale_fill_manual(values = cols) +
  theme(legend.position = "none")
### Save the plot
ggsave(file.path(folder,"nearest-hypermetro.pdf"))







