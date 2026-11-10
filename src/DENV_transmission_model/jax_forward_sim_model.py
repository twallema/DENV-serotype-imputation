import os
import jax
import jax.numpy as jnp
import diffrax
import polars as pl
import pandas as pd
import ast
import numpy as np
from datetime import datetime, timedelta

abs_dir = os.path.dirname(__file__)


class DengueModel:

    def __init__(self, age_groups, municipalities, simulation_start, intro_dates_DENV3, intro_dates_DENV4):
        """
        Initialises the dengue transmission model

        Parameters
        ----------
        age_groups: list[tuple]
            age intervals as (lower_age, upper_age), e.g. [(0, 1), (1, 6), (6, 12), ...]
        
        municipalities: list
            7-digit IBGE municipality codes

        simulation_start: datetime.datetime
            startdate of simulation
            must be first of month

        intro_dates_DENV3/4: list containing datetime.datetime
            per municipality, contains the introduction date of DENV3/4

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
        self.municipalities = municipalities

        # validate simulation start and enddate (current range: 1998-2026)
        simulation_end = datetime(2026, 12, 31)
        assert ((simulation_start.year >= 1998) & (simulation_start.year <= 2026))
        assert (simulation_start - timedelta(days=1)).month != simulation_start.month, "`simulation_start` must fall on the first day of the month."
        assert simulation_end > simulation_start, "`simulation_end` must greater than `simulation_end`"
        self.months = pd.date_range(start=simulation_start, end=simulation_end, freq="ME")

        # load in demographics
        data_path = os.path.join(abs_dir, '../../data/interim/demographics/population_mun-age_1998-2026.parquet')
        self.demographics = self._get_demographics_simstart(data_path, age_groups, municipalities, self.age_groups_labels, simulation_start.year)

        # load in births
        data_path = os.path.join(abs_dir, '../../data/interim/demographics/births_mun_1998-2026.csv')
        self.births = self._get_births(data_path, municipalities, self.months)

        # load in deaths
        data_path = os.path.join(abs_dir, '../../data/interim/demographics/deaths_mun_1998-2026.csv')
        self.deaths = self._get_deaths(data_path, municipalities, self.months, self.age_groups)

        # load in seeding and introduction masks
        data_path = os.path.join(abs_dir, '../../data/interim/datasus_DENV-linelist/DENV-1998_2026-month-mun-age_group-no_diagnostics.parquet')
        self.seeding = self._get_seeding(data_path, municipalities, self.months)
        self.introduction = self._make_introduction_mask(municipalities, self.months, intro_dates_DENV3, intro_dates_DENV4)
        self.seeding_monthly = self.seeding * self.introduction
        self.seeding_monthly /= jnp.sum(self.seeding_monthly, axis=-1)[..., None]   # sum-to-one

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
    def _get_demographics_simstart(data_path, age_groups, municipalities, age_groups_labels, simulation_start_year):
        """
        A function to fetch and format the number of inhabitants in each modeled territory and age group during the simulation start year

        Returns
        -------

        demographics: jnp.ndarray
            shape: (n_munis, n_age_groups)
        """

        breaks = [end for _, end in age_groups[:-1]]

        demo = (
            pl.scan_parquet(data_path)
            .filter(pl.col("CD_MUN").is_in(municipalities))
            .filter(pl.col("year") == simulation_start_year)
            .with_columns(pl.col("age").cut(breaks=breaks, labels=age_groups_labels, left_closed=True).alias("age_group"))
            .group_by([ "CD_MUN", "age_group"]).agg(pl.col("population").sum().alias("population"))
            .sort([ "CD_MUN", "age_group"])
            .collect()
        )

        demo = demo.pivot(values="population", index="CD_MUN", on="age_group", aggregate_function="sum") 

        return jnp.array(demo.select(pl.exclude("CD_MUN")).to_numpy())

    @staticmethod
    def _get_births(data_path, municipalities, months):
        """
        A function that fetches the number of births in each modeled territory

        Returns
        -------

        births: jnp.ndarray
            shape: (n_months, n_munis)
        """

        births = pd.read_csv(data_path)

        births = births[births['CD_MUN'].isin(municipalities)]

        births['births'] /= 12

        # expand to monthly timestep of the model
        monthly_template = pd.MultiIndex.from_product([municipalities, months], names=["CD_MUN", "date"]).to_frame(index=False)
        monthly_template["year"] = monthly_template["date"].dt.year
        births_monthly = pd.merge(monthly_template, births, on=["CD_MUN", "year"], how="left")

        # births (all in first age group) --> (n_months, n_munis)
        births = jnp.array(births_monthly.pivot(index="date", columns="CD_MUN", values="births").to_numpy())

        return births


    @staticmethod
    def _get_deaths(data_path, municipalities, months, age_groups):
        """
        A function that fetches the death rate in each modeled territory

        Assumes a uniform death rate across age groups

        Returns
        -------

        deaths: jnp.ndarray
            shape: (n_months, n_munis, n_age_groups)
        """

        deaths = pd.read_csv(data_path)

        deaths = deaths[deaths['CD_MUN'].isin(municipalities)]

        deaths['death_rate'] = (deaths['deaths'] / 12) / deaths['population']   # compute monthly death rate

        # expand to monthly timestep of the model
        monthly_template = pd.MultiIndex.from_product([municipalities, months], names=["CD_MUN", "date"]).to_frame(index=False)
        monthly_template["year"] = monthly_template["date"].dt.year
        deaths_monthly = pd.merge(monthly_template, deaths, on=["CD_MUN", "year"], how="left")

        deaths_2d = deaths_monthly.pivot(index="date", columns="CD_MUN", values="death_rate").to_numpy()
        deaths = np.repeat(deaths_2d[:,:,None], len(age_groups), axis=-1)

        return jnp.array(deaths)

    @staticmethod
    def _get_seeding(data_path, municipalities, months):
        """
        A function that fetches the incidence data, smooths it and computes the serotype distribution on a rolling window

        Returns
        -------

        estimated_proportions: jnp.ndarray
            shape: (n_months, n_munis, n_serotypes)
        """

        inc = (
            pl.scan_parquet(data_path)
            .filter(pl.col("CD_MUN").is_in(municipalities))
                .filter(pl.col("date").is_in(months))
            .sort([ "date", "CD_MUN"])
            .collect()
        ).to_pandas()

        # Rolling-window version for seeding
        alpha=1/2
        window = 24
        # Aggregate by month and cluster
        cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4"]
        inc_monthly = inc.groupby(['date', 'CD_MUN'])[cols].sum().reset_index()

        # Initialize output array
        rolling_array = np.zeros((len(inc_monthly.date.unique()), len(municipalities), 4))
        # Compute rolling Bayesian for each cluster
        for c_idx, cl in enumerate(municipalities):
            df_cl = inc_monthly[inc_monthly['CD_MUN'] == cl].set_index('date').reindex(months, fill_value=0)[cols]
            for i in range(len(inc_monthly.date.unique())):
                start_idx = max(0, i - window + 1)
                window_sum = df_cl.iloc[start_idx:i+1].sum()
                bayes = (window_sum + alpha) / (window_sum.sum() + 4*alpha)
                rolling_array[i, c_idx, :] = bayes.values
        estimated_proportions = rolling_array   # shape: (n_months, n_clusters, n_serotypes)

        return jnp.array(estimated_proportions)

    @staticmethod
    def _make_introduction_mask(municipalities, months, intro_dates_DENV3, intro_dates_DENV4):
        """
        A function that fetches the incidence data, smooths it and computes the serotype distribution on a rolling window

        Parameters
        ----------

        municipalities: list
            7-digit IBGE municipality codes

        months: DatetimeIndex
            simulation timesteps

        intro_dates_DENV3/4: list containing datetime.datetime
            per municipality, contains the introduction date of DENV3/4

        Returns
        -------

        estimated_proportions: jnp.ndarray
            shape: (n_months, n_munis, n_serotypes)
        """

        assert len(intro_dates_DENV3) == len(intro_dates_DENV4) == len(municipalities)

        intro_mask = np.ones((len(months), len(municipalities), 4))

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
            shape: (n_munis,)
            Annual force of infection for each municipality.
            
        [self] age_groups : list[tuple]
            Age intervals as (lower_age, upper_age), e.g. [(0, 1), (1, 6), (6, 12), ...]

        Returns
        -------
        f_P : jnp.ndarray
            shape: (n_munis, n_age_groups, 3)
            Probability of having 0, 1, or >=2 prior infections for each municipality and age group.
        """
        
        midpoint_age = jnp.array([(lo + hi) / 2 for lo, hi in self.age_groups])

        foi = foi_year[:, None]     # foi: (n_munis, 1)
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
            shape (n_munis, n_age_groups)
            population count

        f_P: jnp.ndarray
            shape (n_munis, n_age_groups)
            fraction of total population in cross-protected state

        pi_d : jnp.ndarray
            shape (n_munis, n_age_groups, 3)
            fraction of (muni,age) in degree of infection: [naive, mono, double]

        pi_mono2 : jnp.ndarray
            shape (n_munis, n_age_groups)
            fraction of mono infected by DENV-2

        Returns
        -------
        S0 : jnp.ndarray
            shape (n_munis, n_age_groups, 16)
            initial S states
        """
        
        z = jnp.zeros_like(pi_d[..., 0])                                   # (n_munis, n_age_groups)
        S_frac = jnp.stack([pi_d[..., 0],
                        pi_d[..., 1] * (1 - pi_mono2),
                        pi_d[..., 1] * pi_mono2,
                        z, z,
                        pi_d[..., 2]] + [z] * 10, axis=-1)             # (n_munis, n_age_groups, 16)
        return (1 - f_P)[..., None] * self.demographics[..., None] * S_frac


    def build_initial_crossprotection(self, f_P, pi_d, f_P2):
        """
        Construct the initial cross-protected state (only DENV-1 and DENV-2 histories).

        Parameters
        ----------
        [from self] demographics : jnp.ndarray
            shape (n_munis, n_age_groups)
            population count

        f_P : jnp.ndarray
            shape (n_munis, n_age_groups)
            fraction of total population in cross-protected state

        pi_d : jnp.ndarray
            shape (n_munis, n_age_groups, 3)
            fraction of (muni,age) in degree of infection: [naive, mono, double]

        f_P2 : jnp.ndarray
            shape (n_munis, n_age_groups)
            fraction of the cross-protected population that is protected after a DENV-2 infection

        Returns
        -------
        P0 : jnp.ndarray
            shape (n_munis, n_age_groups, 32)
        """
        # normalize pi_d over degree of infection 1 and 2
        pi_sum = pi_d[..., 1] + pi_d[..., 2] + 1e-12         # prevent division by zero
        deg1_frac = pi_d[..., 1] / pi_sum                    # (n_munis, n_age_groups)
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
            shape (n_munis, n_age_groups)
            population count

        I_estim: jnp.ndarray
            shape: (n_munis,)
            initial fraction of population infected
            (assumed uniform over age groups)
        
        p0: jnp.ndarray
            shape: (n_clusters, n_serotypes)
            initial serotype distribution 
            
        pi_d : jnp.ndarray
            shape (n_munis, n_age_groups, 3)
            fraction of (muni,age) in degree of infection: [naive, mono, double]

        Returns
        -------
        I0 : jnp.ndarray
            shape (n_munis, n_age_groups, 32)
        """

        # divide the total dengue cases over the serotypes under the assumption they are heterologous infections
        I_estim_star = ((I_estim[:, None] * self.demographics)[:, :, None] * p0[:,None,:])[:, :, None, :] * jnp.stack([ pi_d[..., 0],  pi_d[..., 1],  pi_d[..., 2], jnp.zeros_like(pi_d[..., 0]), jnp.zeros_like(pi_d[..., 0])], axis=-1)[:, :, :, None]
        # shape: n_munis, n_age_groups, n_prior_heterologous infections, n_serotypes

        z = jnp.zeros_like(I_estim_star[:, :, 0, 0])
        return jnp.stack([I_estim_star[:, :, 0, 0], I_estim_star[:, :, 0, 1], I_estim_star[:, :, 0, 2], I_estim_star[:, :, 0, 3], # naive
                            I_estim_star[:, :, 1, 1], I_estim_star[:, :, 1, 2], I_estim_star[:, :, 1, 3],                       # one prior DENV 1
                            I_estim_star[:, :, 1, 0], I_estim_star[:, :, 1, 2], I_estim_star[:, :, 1, 3],                       # one prior DENV 2
                                z, z, z, z, z, z,
                                I_estim_star[:, :, 2, 2], I_estim_star[:, :, 2, 3],                                             # two prior DENV 1 + 2
                                    z, z, z, z, z, z, z, z, z, z, z, z, z, z], axis=-1)


    ##########################
    ## Simulation mechanics ##
    ##########################
