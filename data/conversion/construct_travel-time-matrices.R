library(geobr)
library(brpop)
library(osrm)
library(osrm.backend)
library(dplyr)

# Fetch the OpenStreetMap for Brazil
# https://download.geofabrik.de/south-america/brazil.html --> Download brazil-latest.osm.pbf

geo_level <- "CD_RGINT"
year = 2022
N_runs <- 50 # Number of random municipality pairs to sample per region pair
max_attempts <- 10
travel_duration_threshold <- 999
  
# Set working directory to location of this script
if(!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

###############################
## Start a local OSRM server ##
###############################

osrm_stop()
internal_pbf <- file.path(getwd(), "../raw/travel_time_matrices", "brazil-260729.osm.pbf")
osrm_start(internal_pbf, verbose = FALSE)
options("osrm.server" = "http://localhost:5001/")

######################
## Data Preparation ##
######################

# Load region-to-muni mapping and population data
muni_region_map <- read.csv(file.path(getwd(), "../interim/spatial_units_mapping.csv"))

muni_pop <- ibge_pop() %>% rename(CD_MUN = code_muni) %>% filter(year == "2022")

muni_region_map <- muni_region_map %>% 
  left_join(muni_pop %>% select(c("CD_MUN", "pop")), by = "CD_MUN")

# Compute fraction of RGI/RGINT pop in muni
muni_region_map <- muni_region_map %>%
  
  group_by(CD_RGI) %>%
  mutate(fraction_RGI = pop / sum(pop)) %>%

  group_by(CD_RGINT) %>%
  mutate(fraction_RGINT = pop / sum(pop)) %>%
  ungroup()

# Get the municipality ("seats")
seats <- read_municipal_seat(year=year) %>% rename(CD_MUN = code_muni)

muni_region_map <- muni_region_map %>% 
  left_join(seats %>% select(c("CD_MUN", "geometry")), by = "CD_MUN")

###########################
## Monte-Carlo OSM Query ##
###########################

if (geo_level == "CD_RGI") {
  muni_region_map$region_code <- muni_region_map$CD_RGI
  muni_region_map$pop_fraction  <- muni_region_map$fraction_RGI
  save_name <- "rgi"
} else {
  muni_region_map$region_code <- muni_region_map$CD_RGINT
  muni_region_map$pop_fraction  <- muni_region_map$fraction_RGINT
  save_name <- "rgint"
}

# Get unique region codes for matrix dimensions
regions <- sort(unique(muni_region_map$region_code))
n_regions <- length(regions)

# Initialize matrices for mean and standard deviation
mean_matrix <- matrix(NA, nrow = n_regions, ncol = n_regions, dimnames = list(regions, regions))
sd_matrix   <- matrix(NA, nrow = n_regions, ncol = n_regions, dimnames = list(regions, regions))
max_attempts_reached_matrix <- matrix(0, nrow = n_regions, ncol = n_regions, dimnames = list(regions, regions))

# Pre-filter municipalities by region to optimize sampling performance inside loops
region_muni_list <- split(muni_region_map, muni_region_map$region_code)

# Initialize progress bar
pb <- txtProgressBar(min = 0, max = n_regions, style = 3)

for (i in seq_along(regions)) {
  
  orig_reg <- regions[i]
  orig_pool <- region_muni_list[[as.character(orig_reg)]]
  
  for (j in seq_along(regions)) {
    
    if (i == j) {
      mean_matrix[i, j] <- 0
      sd_matrix[i, j]   <- 0
      max_attempts_reached_matrix[i, j] <- 0
      next
    }
    
    dest_reg <- regions[j]
    dest_pool <- region_muni_list[[as.character(dest_reg)]]
    
    run_durations <- numeric(N_runs)
   
    pair_max_failures <- 0
    
    for (r in 1:N_runs) {
      sampled_orig <- orig_pool[sample(nrow(orig_pool), size = 1, prob = orig_pool$pop_fraction), ]
      sampled_dest <- dest_pool[sample(nrow(dest_pool), size = 1, prob = dest_pool$pop_fraction), ]
      
      if (i == j) {
        run_durations[r] <- 0
        next
      }
      
      # Retry loop until successful response or max retries reached
      success <- FALSE
      attempts <- 0
      
      while (!success && attempts < max_attempts) {
        attempts <- attempts + 1
        
        tryCatch({
          suppressMessages({
            route <- osrmRoute(
              src = sampled_orig$geometry,  
              dst = sampled_dest$geometry,  
              overview = FALSE,  
              osrm.profile = "car"
            )
          })
          
          dur <- route["duration"]
          if (!is.na(dur) && length(dur) > 0) {
            run_durations[r] <- dur / 60
            success <- TRUE # Exit retry loop on success
          } else {
            Sys.sleep(0.01) # Brief pause before retrying
          }
          
        }, error = function(e) {
          # If connection fails, pause briefly and let the while loop try again
          Sys.sleep(0.01)
        })
      }
      
      # If all attempts failed, set as NA and track
      if (!success) {
        run_durations[r] <- NA
        pair_max_failures <- pair_max_failures + 1
      }
    }
    
    # Register number of runs with all attempts failed
    max_attempts_reached_matrix[i, j] <- pair_max_failures / N_runs
    
    # Clean failed runs (NAs)
    valid_durations <- run_durations[!is.na(run_durations)]
    
    # Clean runs greater than a "sanity" threshold
    valid_durations[valid_durations > travel_duration_threshold] <- NA
    
    if (length(valid_durations) > 0) {
      mean_matrix[i, j] <- mean(valid_durations)
      sd_matrix[i, j]   <- sd(valid_durations)
      if (length(valid_durations) == 1) sd_matrix[i, j] <- 0 
    } else {
      mean_matrix[i, j] <- NA
      sd_matrix[i, j]   <- NA
    }
    
  }
  
  # Update progress bar
  setTxtProgressBar(pb, i)
  
}

# Close progress bar connection
close(pb)

# Stop server
osrm_stop()

##################
## Save Results ##
##################

write.csv(as.data.frame(mean_matrix), sprintf("../interim/travel_time_matrices/travel-time_car_mean_%s.csv", save_name))
write.csv(as.data.frame(sd_matrix), sprintf("../interim/travel_time_matrices/travel-time_car_sd_%s.csv", save_name))
write.csv(as.data.frame(max_attempts_reached_matrix), sprintf("../interim/travel_time_matrices/fraction_max-attempts-reached_%s.csv", save_name))
