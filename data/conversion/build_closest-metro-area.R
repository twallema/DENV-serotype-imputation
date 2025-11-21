library(sf)
library(geobr)
library(igraph)
library(ggplot2)
library(tidyverse)
library(randomcoloR)

# Download shape files of states, municipalities
shape_states = geobr::read_state(year=2018)
shape_munis = read_municipality(year=2018)

# Download metro areas and urban areas & visualise on map
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
# Take out imperfections
metro_polys <- metro_polys %>%
  mutate(geometry = st_simplify(geometry, dTolerance = 100))

# Find which metros share a border
touch_list <- st_intersects(metro_polys)
g <- graph_from_adj_list(touch_list)
clusters <- components(g)$membership
metro_polys$cluster_id <- clusters

# Construct hypermetro areas (33)
metro_polys <- metro_polys %>%
  group_by(cluster_id) %>%
  summarise(geometry = st_union(geometry)) %>%
  st_as_sf()
# Take out imperfections
metro_polys <- metro_polys %>%
  mutate(geometry = st_simplify(geometry, dTolerance = 100))

# Compute hypermetro area centroid 
metro_centroids <- metro_polys %>% 
  st_transform(3857)
metro_centroids <- st_centroid(metro_centroids)

# Find out what municipalities are not in a metro area
all_munis <- shape_munis %>% st_transform(3857)
munis_not_metro <- all_munis %>%
  filter(!(code_muni %in% metros$code_muni))

# Find out what municipalities are in a metro area
munis_metro <- all_munis %>%
  filter(code_muni %in% metros$code_muni)

# Compute the distance from each non metro municipality to the hypermetro areas
dist_matrix <- st_distance(munis_not_metro, metro_centroids)

# Find the nearest hypermetro area
nearest_idx <- apply(dist_matrix, 1, which.min)
munis_not_metro$nearest_hypermetro_id <- metro_centroids$cluster_id[nearest_idx]

### Visualise nearest hypermetro areas
n_cols <- length(unique(munis_not_metro$nearest_hypermetro_id))
cols <- randomColor(n_cols, luminosity = "bright")
ggplot() + theme_void() + theme_bw() + 
  ggtitle("Urban clusters") +
  geom_sf(data = shape_states, color="black") +
  geom_sf(data = metro_polys, color='grey') +
  geom_sf(data = munis_not_metro, aes(fill=factor(nearest_hypermetro_id)), alpha=1) +
  scale_fill_manual(values = cols) +
  theme(legend.position = "none") #+
#coord_sf(xlim = c(-50, -40), ylim = c(-25, -20))













