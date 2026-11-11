import os
import jax
import jax.numpy as jnp
import diffrax
import polars as pl
import pandas as pd
import ast
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

abs_dir = os.path.dirname(__file__)


class DengueModel:

    def __init__(self, age_groups, muni_clusters, simulation_start, intro_dates_DENV3, intro_dates_DENV4):
        """
        Initialises the dengue transmission model

        Parameters
        ----------
        age_groups: list[tuple]
            age intervals as (lower_age, upper_age), e.g. [(0, 1), (1, 6), (6, 12), ...]
        
        muni_clusters: pd.DataFrame
            must contain exactly two columns, of which one must be named 'CD_MUN' and the other's name is free to be chosen by the user
            the free column name will be used as the spatial dimension name
            the dataframe must contain a mapping from the desired spatial dimension to the constituent municipalities

        simulation_start: datetime.datetime
            startdate of simulation
            must be first of month

        intro_dates_DENV3/4: list containing datetime.datetime
            per cluster of municipalities, contains the introduction date of DENV3/4

        Returns
        -------

        DengueModel : object
            initialised age-structured dengue model, run independently across several territories
            to run, use `sim()` function
            to format output cleanly, use `` on the simulation output
        """

        # load in the age groups and spatial units
        self.age_groups = age_groups
        self.age_groups_labels = [f"[{start}, {end})" for start, end in age_groups]

        # load in the spatial grouping
        self.spatial_dimension = muni_clusters.columns[muni_clusters.columns != 'CD_MUN'].values[0]
        self.spatial_coordinates = muni_clusters[self.spatial_dimension].unique()
        municipality_mapping = [muni_clusters.loc[muni_clusters[self.spatial_dimension] == coord, 'CD_MUN'].values for coord in self.spatial_coordinates]

        # validate simulation start and enddate (current range: 1998-2026)
        simulation_end = datetime(2026, 12, 31)
        assert ((simulation_start.year >= 1998) & (simulation_start.year <= 2026))
        assert (simulation_start - timedelta(days=1)).month != simulation_start.month, "`simulation_start` must fall on the first day of the month."
        assert simulation_end > simulation_start, "`simulation_end` must greater than `simulation_end`"
        self.months = pd.date_range(start=simulation_start, end=simulation_end, freq="ME")

        # load in demographics
        data_path = os.path.join(abs_dir, '../../data/interim/demographics/population_mun-age_1998-2026.parquet')
        self.demographics = self._get_demographics_simstart(data_path, age_groups, municipality_mapping, self.age_groups_labels, simulation_start.year)

        # load in births
        data_path = os.path.join(abs_dir, '../../data/interim/demographics/births_mun_1998-2026.csv')
        self.births = self._get_births(data_path, municipality_mapping, self.months)

        # load in deaths
        data_path = os.path.join(abs_dir, '../../data/interim/demographics/deaths_mun_1998-2026.csv')
        self.death_rate = self._get_deaths(data_path, municipality_mapping, self.months, self.age_groups)

        # load in seeding and introduction masks
        data_path = os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/DENV-1998_2026-month-mun-age_group-no_diagnostics.parquet')
        self.seeding = self._get_seeding(data_path, municipality_mapping, self.months)
        self.introduction = self._make_introduction_mask(municipality_mapping, self.months, intro_dates_DENV3, intro_dates_DENV4)
        self.seeding *= self.introduction
        self.seeding /= jnp.sum(self.seeding, axis=-1)[..., None]   # sum-to-one

        # load in the transitions between states
        S_mapping, IP_mapping = self._get_raw_compartmental_flows(os.path.join(abs_dir, '../../data/raw/transmission_model'))
        self.A, self.C, self.W, self.L, self.H_het, self.H_hom, self.K_het, self.K_hom, self.birth_vec = self._flows_to_transition_matrices(S_mapping, IP_mapping, age_groups, 4)
        self.O, self.R_het, self.R_hom, D = self._flows_to_post_simulation_state_mappings(S_mapping, IP_mapping, 4)

        # --> remaining for user: omega, gamma, beta_monthly, p_hom_monthly, S0, I0, P0
    
        pass


    #####################################
    ## Initialisation helper functions ##
    #####################################

    @staticmethod
    def _get_demographics_simstart(data_path, age_groups, muni_mapping, age_groups_labels, simulation_start_year):
        """
        A function to fetch and format the number of inhabitants in each modeled territory and age group during the simulation start year

        Returns
        -------

        demographics: jnp.ndarray
            shape: (n_clusters, n_age_groups)
        """

        breaks = [end for _, end in age_groups[:-1]]

        demo = (
            pl.scan_parquet(data_path)
            .filter(pl.col("year") == simulation_start_year)
            .with_columns(pl.col("age").cut(breaks=breaks, labels=age_groups_labels, left_closed=True).alias("age_group"))
            .group_by([ "CD_MUN", "age_group"]).agg(pl.col("population").sum().alias("population"))
            .sort([ "CD_MUN", "age_group"])
            .collect()
            .to_pandas()
        )

        demo = demo.pivot(values="population", index="CD_MUN", columns="age_group") 

        return jnp.array([demo.loc[group].sum(axis=0).to_numpy() for group in muni_mapping])    # sum across munis in cluster

    @staticmethod
    def _get_births(data_path, muni_mapping, months):
        """
        A function that fetches the number of births in each modeled territory

        Returns
        -------

        births: jnp.ndarray
            shape: (n_months, n_clusters)
        """

        births = pd.read_csv(data_path)

        mun_to_group = {int(mun): idx for idx, group in enumerate(muni_mapping) for mun in group} # create mapping: {CD_MUN: group_index}

        births = (
            births.assign(group_id=births["CD_MUN"].map(mun_to_group))
            .dropna(subset=["group_id"])
            .astype({"group_id": int})
            .groupby(["group_id", "year"], as_index=False)[["births", "population"]]
            .sum()
        )

        births['births'] /= 12

        # expand to monthly timestep of the model
        monthly_template = pd.MultiIndex.from_product([births['group_id'].unique(), months], names=["group_id", "date"]).to_frame(index=False)
        monthly_template["year"] = monthly_template["date"].dt.year
        births_monthly = pd.merge(monthly_template, births, on=["group_id", "year"], how="left")

        return jnp.array(births_monthly.pivot(index="date", columns="group_id", values="births").to_numpy())


    @staticmethod
    def _get_deaths(data_path, muni_mapping, months, age_groups):
        """
        A function that fetches the death rate in each modeled territory

        Assumes a uniform death rate across age groups

        Returns
        -------

        deaths: jnp.ndarray
            shape: (n_months, n_clusters, n_age_groups)
        """

        deaths = pd.read_csv(data_path)

        mun_to_group = {int(mun): idx for idx, group in enumerate(muni_mapping) for mun in group} # create mapping: {CD_MUN: group_index}

        deaths = (
            deaths.assign(group_id=deaths["CD_MUN"].map(mun_to_group))
            .dropna(subset=["group_id"])
            .astype({"group_id": int})
            .groupby(["group_id", "year"], as_index=False)[["deaths", "population"]]
            .sum()
        )

        deaths['death_rate'] = (deaths['deaths'] / 12) / deaths['population']   # compute monthly death rate

        # expand to monthly timestep of the model
        monthly_template = pd.MultiIndex.from_product([deaths['group_id'].unique(), months], names=["group_id", "date"]).to_frame(index=False)
        monthly_template["year"] = monthly_template["date"].dt.year
        death_rate_monthly = pd.merge(monthly_template, deaths, on=["group_id", "year"], how="left")

        # repeat across age groups
        deaths_2d = death_rate_monthly.pivot(index="date", columns="group_id", values="death_rate").to_numpy()
        deaths = np.repeat(deaths_2d[:,:,None], len(age_groups), axis=-1)

        return jnp.array(deaths)

    @staticmethod
    def _get_seeding(data_path, muni_mapping, months):
        """
        A function that fetches the incidence data, smooths it and computes the serotype distribution on a rolling window

        Returns
        -------

        estimated_proportions: jnp.ndarray
            shape: (n_months, n_clusters, n_serotypes)
        """

        inc = (
            pl.scan_parquet(data_path)
            .filter(pl.col("date").is_in(months))
            .sort([ "date", "CD_MUN"])
            .collect()
        ).to_pandas()

        mun_to_group = {int(mun): idx for idx, group in enumerate(muni_mapping) for mun in group} # create mapping: {CD_MUN: group_index}

        inc = (
            inc.assign(group_id=inc["CD_MUN"].map(mun_to_group))
            .dropna(subset=["group_id"])
            .astype({"group_id": int})
            .groupby(["group_id", "date"], as_index=False)[["DENV_1", "DENV_2", "DENV_3", "DENV_4"]]
            .sum()
        )

        # Rolling-window version for seeding
        alpha=1/2
        window = 24

        # Initialize output array
        rolling_array = np.zeros((len(inc.date.unique()), len(muni_mapping), 4))
        # Compute rolling Bayesian for each cluster
        for c_idx, cl in enumerate(inc['group_id'].unique()):
            df_cl = inc[inc['group_id'] == cl].set_index('date').reindex(months, fill_value=0)[["DENV_1", "DENV_2", "DENV_3", "DENV_4"]]
            for i in range(len(inc.date.unique())):
                start_idx = max(0, i - window + 1)
                window_sum = df_cl.iloc[start_idx:i+1].sum()
                bayes = (window_sum + alpha) / (window_sum.sum() + 4*alpha)
                rolling_array[i, c_idx, :] = bayes.values
        estimated_proportions = rolling_array   # shape: (n_months, n_clusters, n_serotypes)

        return jnp.array(estimated_proportions)

    @staticmethod
    def _make_introduction_mask(muni_mapping, months, intro_dates_DENV3, intro_dates_DENV4):
        """
        A function that fetches the incidence data, smooths it and computes the serotype distribution on a rolling window

        Parameters
        ----------

        muni_mapping: list containing lists
            1st level: municipality clusters, 2nd level: 7-digit IBGE code for municipalities in cluster

        months: DatetimeIndex
            simulation timesteps

        intro_dates_DENV3/4: list containing datetime.datetime
            per cluster of municipalities, contains the introduction date of DENV3/4

        Returns
        -------

        estimated_proportions: jnp.ndarray
            shape: (n_months, n_clusters, n_serotypes)
        """

        assert len(intro_dates_DENV3) == len(intro_dates_DENV4) == len(muni_mapping)

        intro_mask = np.ones((len(months), len(muni_mapping), 4))

        for i, (intro_date_DENV3, intro_date_DENV4) in enumerate(zip(intro_dates_DENV3, intro_dates_DENV4)):
            intro_mask[:, i, 2] = (months >= intro_date_DENV3).astype(int)
            intro_mask[:, i, 3] = (months >= intro_date_DENV4).astype(int)

        return jnp.array(intro_mask)

    @staticmethod
    def _get_raw_compartmental_flows(data_path):
        """
        Finds and formats the raw structure of the transmission model (flows between compartments)
        """

        S_mapping = pd.read_csv(os.path.join(data_path, 'susceptible_states.csv'))
        list_columns = ["susc_to_heterol_inf", "susc_to_homol_inf"]
        
        for col in list_columns:
            S_mapping[col] = S_mapping[col].apply(ast.literal_eval) # Convert strings to actual lists

        IP_mapping = pd.read_csv(os.path.join(data_path, 'infectious_protected_states.csv'))

        return S_mapping, IP_mapping
    
    @staticmethod
    def _flows_to_transition_matrices(S_mapping, IP_mapping, age_groups, n_serotypes):
        """
        Transforms the flows between compartments into the transition matrices
        """

        # define lengths
        n_I = len(IP_mapping)
        n_S = len(S_mapping)
        n_age_groups = len(age_groups)
        sero_idx = IP_mapping["currently_infected_with"].values - 1

        # births
        birth_vec = jnp.zeros([n_age_groups, n_S])
        birth_vec = birth_vec.at[0,0].set(1.0)

        # C --> Maps infected states onto serotypes
        C = np.zeros((n_I, n_serotypes))
        C[np.arange(n_I), sero_idx] = 1.0
        C = jnp.array(C)

        # W --> Maps protected states waning to S
        W = jnp.array(np.eye(n_S)[IP_mapping["P_to_S"]])

        # Susceptibility of S states to serotypes (M = H_het + f * H_hom)
        H_het = jnp.zeros((n_S, n_serotypes))
        H_hom = jnp.zeros((n_S, n_serotypes))
        for j, row in S_mapping.iterrows():
            for k in row["susc_to_heterol_inf"]:
                H_het = H_het.at[j, int(k)-1].set(1.0)
            for k in row["susc_to_homol_inf"]:
                H_hom = H_hom.at[j, int(k)-1].set(1.0)
        H_het = jnp.transpose(H_het)
        H_hom = jnp.transpose(H_hom)

        # Flow of S states into I states (heterologous and homologous)
        K_het = jnp.zeros((n_I, n_S))
        K_hom = jnp.zeros((n_I, n_S))
        for i, row in IP_mapping.iterrows():
            K_het = K_het.at[i, int(row["S_to_I_heterol"])].set(1.0)
            K_hom = K_hom.at[i, int(row["S_to_I_homol"])].set(1.0)
        K_het = jnp.transpose(K_het)
        K_hom = jnp.transpose(K_hom)

        # Map serotype-specific FOI to the per I state FOI
        L = jnp.zeros((n_serotypes, n_I))
        for i, s in enumerate(IP_mapping["currently_infected_with"].values - 1):
            L = L.at[s, i].set(1.0)


        # aging matrix (n_age_groups x n_age_groups)

        ## compute transition rates based on age group sizes
        widths_months = np.array([(b[1] - b[0]) * 12 for b in age_groups])
        aging_rates = np.zeros(len(age_groups))
        aging_rates[:-1] = 1.0 / widths_months[:-1]

        ## build the transition matrix (n_groups x n_groups)
        A_aging = np.zeros((len(age_groups), len(age_groups)))
        np.fill_diagonal(A_aging, -aging_rates) # negative outflows on the diagonal
        for i in range(len(age_groups) - 1):
            A_aging[i + 1, i] = aging_rates[i] # positive inflows on the sub-diagonal

        # convert to jnp array
        A = jnp.array(A_aging)

        return A, C, W, L, H_het, H_hom, K_het, K_hom, birth_vec

    @staticmethod
    def _flows_to_post_simulation_state_mappings(S_mapping, IP_mapping, n_serotypes):
        """
        Transforms the flows between compartments into post-simulation mappings of the raw S, I and P simulation states into more useful objects for downstream evaluation
        """

        ## define lengths
        n_I = len(IP_mapping)
        n_S = len(S_mapping)
    
        # post-hoc observation mapping
        ## Delta_I mapping to (cluster, degree, serotype)
        O = jnp.zeros((n_I, IP_mapping["no_prior_heterol_inf"].max() + 1, n_serotypes))
        for i, row in IP_mapping.iterrows():
            d = int(row["no_prior_heterol_inf"])
            s = int(row["currently_infected_with"]) - 1
            O = O.at[i, d, s].set(1.0)

        ## S (slots) mapping to (cluster, degree, serotype)
        ### make two maps: one to heterologous and one to homologous 
        ### susceptibility slots = heterologous + f_i * homologous
        R_het = jnp.zeros((n_S, S_mapping["no_prior_heterol_inf"].max() + 1, n_serotypes))
        R_hom = jnp.zeros((n_S, S_mapping["no_prior_heterol_inf"].max() + 1, n_serotypes))
        for i, row in S_mapping.iterrows():
            d = int(row["no_prior_heterol_inf"])
            for s in row["susc_to_heterol_inf"]:
                R_het = R_het.at[i, d, s-1].set(1.0)
            for s in row["susc_to_homol_inf"]:
                R_hom = R_hom.at[i, d, s-1].set(1.0)

        ## S (individuals) mapping to (cluster, degree)
        D = jnp.zeros((n_S, S_mapping["no_prior_heterol_inf"].max() + 1))
        for i, row in S_mapping.iterrows():
            D = D.at[i, int(row["no_prior_heterol_inf"])].set(1.0)

        return O, R_het, R_hom, D
    

    ##########################
    ## Simulation mechanics ##
    ##########################

    @staticmethod
    def RHS(t, y, args):
        """
        Continuous-time derivatives for one spatial unit across age groups.

        Inputs: y, args (see `simulate_one_jax`)
        Returns dS_dt (n_age_groups, n_S), dI_dt (n_age_groups, n_I), dP_dt (n_age_groups, n_P),
                dI_hom (n_age_groups, n_I), dI_het (n_age_groups, n_I)
        """
        
        # unpack states
        S_t, I_t, P_t, _, _ = y
        
        # unpack parameters
        gamma, omega, beta_monthly, births_monthly, death_rate_monthly, p_hom_monthly, seeding_monthly, A, C, W, L, H_het, H_hom, K_het, K_hom, birth_vec, ts = args
        
        # look up beta, births and deaths, and seeding (jnp.interp)
        beta_t = jnp.interp(t, xp=ts, fp=beta_monthly)
        births_t = jnp.interp(t, xp=ts, fp=births_monthly)
        deaths_t = jax.vmap(lambda x: jnp.interp(t, ts, x))(death_rate_monthly.T)
        p_hom_t = jax.vmap(lambda x: jnp.interp(t, ts, x))(p_hom_monthly.T)
        seeding_t = jax.vmap(lambda x: jnp.interp(t, ts, x))(seeding_monthly.T)
        
        # --- FOI: shared across age groups --------------------------------------------
        N_t = S_t.sum() + I_t.sum() + P_t.sum()                                    # scalar
        I_per_sero = I_t.sum(axis=0) @ C                                           # (n_serotypes,)
        lambda_t = beta_t * I_per_sero / N_t                                       # (n_serotypes,)
        lambda_t = lambda_t + 1e-7 * seeding_t                                     # (n_serotypes)

        # --- Susceptible --------------------------------------------------------------
        effective_lambda = lambda_t @ H_het + (lambda_t * p_hom_t) @ H_hom             # (n_S,)
        dS_dt = (births_t * birth_vec
                - deaths_t[:, None] * S_t
                + (1.0 / omega) * (P_t @ W)
                - effective_lambda[None, :] * S_t
                + A @ S_t)                                                        # (n_age_groups, n_S)
    
        # --- Infectious ---------------------------------------------------------------
        f_per_I = p_hom_t @ L                                                      # (n_I,)
        lambda_per_I = lambda_t @   L                                              # (n_I,)
        dI_het_dt = lambda_per_I[None, :] * (S_t @   K_het)                        # (n_age_groups, n_I)
        dI_hom_dt = lambda_per_I[None, :] * f_per_I[None, :] * (S_t @ K_hom)       # (n_age_groups, n_I)
        dI_dt = (- deaths_t[:, None] * I_t
                - (1 / gamma) * I_t
                + dI_het_dt + dI_hom_dt
                + A @ I_t)

        # --- Cross-protected ----------------------------------------------------------
        dP_dt = (- (1 / omega) * P_t
                - deaths_t[:, None] * P_t
                + (1 / gamma) * I_t
                + A @ P_t)

        return dS_dt, dI_dt, dP_dt, dI_hom_dt, dI_het_dt


    def simulate_jax_single(self,
                            gamma, omega,
                            A, C, W, L, H_het, H_hom, K_het, K_hom, birth_vec, 
                            beta_monthly, births_monthly, death_rate_monthly, p_hom_monthly, seeding_monthly,
                            S0, I0, P0,
                            ts, stepsize):
        
        """
        Simulate the age-structured dengue transmission model.

        Uses Bosh3 with a fixed step size

        Parameters
        ----------
        gamma: scalar
            Duration of infection.
            
        omega: scalar
            Duration of cross protection.
            
        A: jnp.ndarray
            (n_age_groups, n_age_groups)
            Aging transition matrix.
            
        C: jnp.ndarray
            (32, n_serotypes)
            Maps infected states onto serotypes.

        W: jnp.ndarray
            (32, 16)
            Maps how P states wane to S states.

        L: jnp.ndarray
            (4, 32)
            Map serotype-specific FOI to the per I state FOI.

        H_het / H_hom: jnp.ndarray
            (16, 4)
            Susceptibility of S states to serotypes.
                
        K_het / K_hom : jnp.ndarray
            (16, 32)
            Flow of S states into I states.
                
        birth_vec: jnp.ndarray
            (n_age_groups, 16)
            All elements zero except at index (0,0).
            
        beta_monthly: jnp.ndarray
            (n_months,)
            Time-varying transmission rate.
            
        births_monthly: jnp.ndarray
            (n_months,)
            Time-varying number of births.
            
        death_rate_monthly: jnp.ndarray
            (n_months,)
            Time-varying number of deaths.
            
        p_hom_monthly: jnp.ndarray
            (n_months, n_age_groups, n_serotypes)
            Time-varying homologous reinfection probability.
            
        seeding_monthly: jnp.ndarray
            (n_months, n_age_groups, n_serotypes)
            Time-varying seeding.
            
        S0 : jnp.ndarray
            (n_age_groups, 16)
            Initial susceptibles.
            
        I0 : jnp.ndarray
            (n_age_groups, 32)
            Initial infected.

        P0: jnp.ndarray
            (n_age_groups, 32)
            Initial cross-protected.

        ts : array
            Simulation and observation dates.

        stepsize: float
            Simulation stepsize (fixed)

        Returns
        -------
        sol.ys : tuple containing jnp.ndarray
            0: S_t (n_months, n_age_groups, 16)
            1: I_t (n_months, n_age_groups, 32)
            2: P_t (n_months, n_age_groups, 32)
            3: C_het_t (n_months, n_age_groups, 32) --> cumulative of dI_het_t
            4: C_hom_t (n_months, n_age_groups, 32) --> cumulative of dI_hom_t
        """

        # Create diffrax term
        term = diffrax.ODETerm(self.RHS)

        # Solve
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Bosh3(),
            t0=ts[0],
            t1=ts[-1],
            dt0=stepsize,
            y0=(S0, I0, P0, jnp.zeros_like(I0), jnp.zeros_like(I0)),
            args=(gamma, omega, beta_monthly, births_monthly, death_rate_monthly, p_hom_monthly, seeding_monthly, A, C, W, L, H_het, H_hom, K_het, K_hom, birth_vec, ts),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.ConstantStepSize(),
            adjoint=diffrax.DirectAdjoint(),
        )
        
        return sol.ys


    def batch_jax_single(self, static_pars, dynamic_pars, initial_states, model_structure_pars, solver_pars):
        
        """
        Batch the age-structured dengue transmission model across spatial units using jax.vmap

        Parameters
        ----------
        see `simulate_jax_single`

        Returns
        -------
        sol.ys : tuple containing jnp.ndarray
            0: S_t (n_months, n_clusters, n_age_groups, 16)
            1: I_t (n_months, n_clusters, n_age_groups, 32)
            2: P_t (n_months, n_clusters, n_age_groups, 32)
            3: C_het_t (n_months, n_clusters, n_age_groups, 32)
            4: C_hom_t (n_months, n_clusters, n_age_groups, 32)
        """

        # define where to find spatial axis
        in_axes = (
            *([None] * len(static_pars)),           # static_pars map to None (no spatial axis)
            *([None] * len(model_structure_pars)),  # model_structure_pars map to None (no spatial axis)
            *([1] * len(dynamic_pars)),             # dynamic_pars map to 1 (spatial axis in position 1)
            *([0] * len(initial_states)),           # initial_states map to 0 (spatial axis in position 0)
            *([None] * len(solver_pars))            # solver_pars map to None (no spatial axis)
        )

        simulation = jax.vmap(self.simulate_jax_single, in_axes=in_axes)
        
        return simulation(*static_pars.values(), *model_structure_pars.values(), *dynamic_pars.values(), *initial_states.values(), *solver_pars.values())


    def sim(self,
            kappa,
            gamma, omega,
            beta_monthly, p_hom_monthly,
            S0, I0, P0,
            simulation_end_date, stepsize):
        
        """
        Simulates the age-structured dengue transmission model across spatial units
        
        Parameters
        ----------
        see `simulate_jax_single`

        Returns
        -------
        #TODO: S_t, I_t, P_t, dI_hom_t, dI_het_t, dI_expanded, S_expanded, dI_obs, p_obs, p_true
        """

        assert ((stepsize > 0) & (stepsize <= 0.5)), "`stepsize` must be in interval )0, 0.5]"
        assert simulation_end_date <= self.months[-1], f"`simulation_end_date` must be smaller than {self.months[-1].strftime("%Y-%m-%d")}."

        # find index of simulation_end_date along date axis --> cut births, deaths, seeding
        idx = self.months.searchsorted(simulation_end_date, side='right')
        assert beta_monthly.shape[0] == idx, f"length of date dimension on `beta_monthly` ({beta_monthly.shape[0]}) must match the number of requested simulated months ({idx})"
        assert p_hom_monthly.shape[0] == idx, f"length of date dimension on `p_hom_monthly` ({beta_monthly.shape[0]}) must match the number of requested simulated months ({idx})"
        
        # divide parameters into functional groups
        static_pars = {'gamma': gamma, 'omega': omega}
        dynamic_pars = {'beta_monthly': beta_monthly, 'births_monthly': self.births[:idx,...], 'death_rate_monthly': self.death_rate[:idx,...],
                        'p_hom_monthly': p_hom_monthly, 'seeding_monthly': self.seeding[:idx,...]}
        initial_states = {'S0': S0, 'I0': I0, 'P0': P0}
        model_structure_pars = {'A': self.A, 'C': self.C, 'W': self.W, 'L': self.L, 'H_het': self.H_het, 'H_hom': self.H_hom,
                                'K_het': self.K_het, 'K_hom': self.K_hom, 'birth_vec': self.birth_vec}
        solver_pars = {'ts': jnp.arange(idx), 'stepsize': stepsize}
        
        # call batch_jax_single to run the ODE
        outputs = self.batch_jax_single(static_pars, dynamic_pars, initial_states, model_structure_pars, solver_pars)
        
        # put time axis on simulation output first
        S_t, I_t, P_t, C_hom_t, C_het_t = jax.tree.map(lambda x: jnp.moveaxis(x, source=1, destination=0), outputs)
        
        # differentiate cumulative infected arrays using padding
        dI_hom_t, dI_het_t = [
            jnp.pad(jnp.diff(C, axis=0), pad_width=((1, 0), *[(0, 0)] * (C.ndim - 1)), mode='constant',  constant_values=0)
            for C in (C_hom_t, C_het_t)
        ]

        # compute derived simulation products
        ## reconstruct the FOI
        N = S_t.sum(axis=(-2, -1)) + I_t.sum(axis=(-2, -1)) + P_t.sum(axis=(-2, -1)) # total population across disease state, age group
        I_per_sero = I_t.sum(axis=2) @ self.C # total infectious population by serotype
        lambda_t = beta_monthly[..., None] * I_per_sero / N[..., None] # force of infection
        
        ## expand the S and I states
        ### reshape dI_het (time, cluster, age, I_state) --> (time, cluster, age_group, infection_degree, serotype, hom/het infection) 
        dI_expanded = jax.nn.softplus(jnp.stack([jnp.einsum("tcai,ids->tcads", dI_hom_t, self.O), jnp.einsum("tcai,ids->tcads", dI_het_t, self.O)], axis=-1))
        
        ### reshape S into susceptibility slots per (time, cluster, age_group, infection_degree, serotype, hom/het infection)
        S_expanded = jnp.stack([jnp.einsum("tcai,ids->tcads", S_t, self.R_hom), jnp.einsum("tcai,ids,tcs->tcads", S_t, self.R_het, p_hom_monthly)], axis=-1)
        
        ## compute observations
        ### compute observed number of cases per serotype
        dI_obs_sero = jnp.sum(dI_expanded * kappa[None,][...,None], axis=(-1, -3)) # (time, cluster, age_group, serotype)
        dI_obs = jnp.sum(dI_expanded * kappa[None,][...,None], axis=(-1, -2, -3)) # (time, cluster, age_group)
        
        ### compute observed serotyped proportions (time, cluster, serotype)
        p_obs = jnp.sum(dI_obs_sero, axis=-2) / jnp.sum(dI_obs_sero, axis=(-2,-1))[..., None]

        ### compute "true" serotype proportions (time, cluster, serotype)
        p_true = I_per_sero / jnp.sum(I_per_sero, axis=-1)[:, :, None]
        
        return S_t, I_t, P_t, dI_hom_t, dI_het_t, dI_expanded, S_expanded, dI_obs, p_obs, p_true, lambda_t


    def simout_to_xarray(self, simout):
        """
        Wrap the simulation output in an xarray with labeled dimensions and coordinates

        Parameters
        ----------

        simout: tuple
            contains raw simulation output ordered:
            S, I, P, dI_hom, dI_het, dI_expanded, S_expanded, dI_obs, p_obs, p_true, lambda_t

        Returns
        -------

        simout: xarray
            formatted simulation output
        """

        _, _, _, _, _, dI_expanded, S_expanded, dI_obs, p_obs, p_true, lambda_t = simout

        len_sim = lambda_t.shape[0]

        simout = xr.Dataset(
            {
                # expanded states
                "S_expanded": (
                    ["date", self.spatial_dimension, "age_group", "n_prior_het_infections", "serotype", "infection_type"],
                    S_expanded,
                ),    
                "dI_expanded": (
                    ["date", self.spatial_dimension, "age_group", "n_het_infections", "serotype", "infection_type"],
                    dI_expanded,
                ),          
                # latent derived states
                "lambda_t": (
                    ["date", self.spatial_dimension, "serotype"],
                    lambda_t
                ),
                "p_true": (
                    ["date", self.spatial_dimension, "serotype"],
                    p_true
                ),
                # observations
                "dI_obs": (
                    ["date", self.spatial_dimension, "age_group"],
                    dI_obs
                ),   
                "p_obs": (
                    ["date", self.spatial_dimension, "serotype"],
                    p_obs
                ),        
            },
            coords={
                self.spatial_dimension: self.spatial_coordinates,
                "date": self.months[:len_sim],
                "age_group": self.age_groups_labels,
                "serotype": [1,2,3,4],
                "n_prior_het_infections": [0, 1, 2, 3, 4],
                "n_het_infections": [1, 2, 3, 4],
                "infection_type": ['hom','het']
            },
        )

        return simout


    ####################################
    ## Initial state helper functions ##
    ####################################

    def infection_history_from_foi(self, foi_year):
        """
        Compute age- and municipality-specific infection-history probabilities
        from a constant force of infection per municipality.

        Parameters
        ----------
        foi_year : jnp.ndarray
            shape: (n_clusters,)
            Annual force of infection for each municipality.
            
        [self] age_groups : list[tuple]
            Age intervals as (lower_age, upper_age), e.g. [(0, 1), (1, 6), (6, 12), ...]

        Returns
        -------
        f_P : jnp.ndarray
            shape: (n_clusters, n_age_groups, 3)
            Probability of having 0, 1, or >=2 prior infections for each municipality and age group.
        """
        
        midpoint_age = jnp.array([(lo + hi) / 2 for lo, hi in self.age_groups])

        foi = foi_year[:, None]     # foi: (n_clusters, 1)
        age = midpoint_age[None, :] # age: (1, n_age_groups)

        # Poisson survival process given FOI
        cumulative_foi = foi * age
        P0 = jnp.exp(-cumulative_foi)
        P1 = cumulative_foi * jnp.exp(-cumulative_foi)
        P2plus = 1.0 - P0 - P1

        return jnp.stack([P0, P1, P2plus], axis=-1)


    def build_initial_susceptibles(self, f_P, pi_d, pi_mono2):
        """
        Construct initial susceptibles for 1999 (only DENV-1 and DENV-2)

        Parameters
        ----------
        [from self] demographics : jnp.ndarray
            shape (n_clusters, n_age_groups)
            population count

        f_P: jnp.ndarray
            shape (n_clusters, n_age_groups)
            fraction of total population in cross-protected state

        pi_d : jnp.ndarray
            shape (n_clusters, n_age_groups, 3)
            fraction of (muni,age) in degree of infection: [naive, mono, double]

        pi_mono2 : jnp.ndarray
            shape (n_clusters, n_age_groups)
            fraction of mono infected by DENV-2

        Returns
        -------
        S0 : jnp.ndarray
            shape (n_clusters, n_age_groups, 16)
            initial S states
        """
        
        z = jnp.zeros_like(pi_d[..., 0])                                   # (n_clusters, n_age_groups)
        S_frac = jnp.stack([pi_d[..., 0],
                        pi_d[..., 1] * (1 - pi_mono2),
                        pi_d[..., 1] * pi_mono2,
                        z, z,
                        pi_d[..., 2]] + [z] * 10, axis=-1)             # (n_clusters, n_age_groups, 16)
        return (1 - f_P)[..., None] * self.demographics[..., None] * S_frac


    def build_initial_crossprotection(self, f_P, pi_d, f_P2):
        """
        Construct the initial cross-protected state (only DENV-1 and DENV-2 histories).

        Parameters
        ----------
        [from self] demographics : jnp.ndarray
            shape (n_clusters, n_age_groups)
            population count

        f_P : jnp.ndarray
            shape (n_clusters, n_age_groups)
            fraction of total population in cross-protected state

        pi_d : jnp.ndarray
            shape (n_clusters, n_age_groups, 3)
            fraction of (muni,age) in degree of infection: [naive, mono, double]

        f_P2 : jnp.ndarray
            shape (n_clusters, n_age_groups)
            fraction of the cross-protected population that is protected after a DENV-2 infection

        Returns
        -------
        P0 : jnp.ndarray
            shape (n_clusters, n_age_groups, 32)
        """
        # normalize pi_d over degree of infection 1 and 2
        pi_sum = pi_d[..., 1] + pi_d[..., 2] + 1e-12         # prevent division by zero
        deg1_frac = pi_d[..., 1] / pi_sum                    # (n_clusters, n_age_groups)
        deg2_frac = pi_d[..., 2] / pi_sum                    

        # P states: 0 = P_1^∅, 1 = P_2^∅, 4 = P_2^1, 7 = P_1^2; all other states start empty
        z = jnp.zeros_like(f_P2)
        P_frac = jnp.stack([deg1_frac * (1 - f_P2), deg1_frac * f_P2, z, z,
                            deg2_frac * f_P2, z, z, deg2_frac * (1 - f_P2)]
                        + [z] * 24, axis=-1)              

        return (f_P * self.demographics)[..., None] * P_frac


    def build_initial_infected(self, I_estim, p0, pi_d):
        """
        Construct the initial infected state (only DENV-1 and DENV-2 histories).
        
        Parameters
        ----------

        [from self] demographics : jnp.ndarray
            shape (n_clusters, n_age_groups)
            population count

        I_estim: jnp.ndarray
            shape: (n_clusters,)
            initial fraction of population infected
            (assumed uniform over age groups)
        
        p0: jnp.ndarray
            shape: (n_clusters, n_serotypes)
            initial serotype distribution 
            
        pi_d : jnp.ndarray
            shape (n_clusters, n_age_groups, 3)
            fraction of (muni,age) in degree of infection: [naive, mono, double]

        Returns
        -------
        I0 : jnp.ndarray
            shape (n_clusters, n_age_groups, 32)
        """

        # divide the total dengue cases over the serotypes under the assumption they are heterologous infections
        I_estim_star = ((I_estim[:, None] * self.demographics)[:, :, None] * p0[:,None,:])[:, :, None, :] * jnp.stack([ pi_d[..., 0],  pi_d[..., 1],  pi_d[..., 2], jnp.zeros_like(pi_d[..., 0]), jnp.zeros_like(pi_d[..., 0])], axis=-1)[:, :, :, None]
        # shape: n_clusters, n_age_groups, n_prior_heterologous infections, n_serotypes

        z = jnp.zeros_like(I_estim_star[:, :, 0, 0])
        return jnp.stack([I_estim_star[:, :, 0, 0], I_estim_star[:, :, 0, 1], I_estim_star[:, :, 0, 2], I_estim_star[:, :, 0, 3], # naive
                            I_estim_star[:, :, 1, 1], I_estim_star[:, :, 1, 2], I_estim_star[:, :, 1, 3],                       # one prior DENV 1
                            I_estim_star[:, :, 1, 0], I_estim_star[:, :, 1, 2], I_estim_star[:, :, 1, 3],                       # one prior DENV 2
                                z, z, z, z, z, z,
                                I_estim_star[:, :, 2, 2], I_estim_star[:, :, 2, 3],                                             # two prior DENV 1 + 2
                                    z, z, z, z, z, z, z, z, z, z, z, z, z, z], axis=-1)


