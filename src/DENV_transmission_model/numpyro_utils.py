"""
This script contains all functions related to the Bayesian numpyro model

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

import os
import polars as pl


def get_incidence_data(data_dir, age_groups):

    labels = [f"[{start}, {end})" for start, end in age_groups]
    breaks = [b[0] for b in age_groups] + [age_groups[-1][1]]

    print(breaks)

    #inc = (
    #    pl.scan_parquet(os.path.join(data_dir, "*.parquet"))
    #    .collect(engine="streaming")
    #)

    print(inc)

    pass

