import os
import jax
import jax.numpy as jnp
import diffrax
import polars as pl
import pandas as pd
import ast
import numpy as np

abs_dir = os.path.dirname(__file__)

###############################
## Transmission model object ##
###############################

class DengueModel:

    def __init__(self, age_groups, municipalities, simulation_start, simulation_end):
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
            # TODO: must be first of month

        simulation_end: datetime.datetime
            enddate of simulation
            # TODO: must be last day of month

        Returns
        -------

        DengueModel : object
            initialised age-structured dengue model, run independently across several territories
            to run, use `sim()` function
            to format output cleanly, use `` on the simulation output
        """

        # load in the age groups, spatial units, simulation start and end --> compute ts
        self.age_groups = age_groups
        self.age_groups_labels = [f"[{start}, {end})" for start, end in age_groups]

        self.municipalities = municipalities

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

        # load in the transitions between states
        S_mapping, IP_mapping = self._get_raw_compartmental_flows(os.path.join(abs_dir, '../../data/raw/transmission_model'))
        self.A, self.C, self.W, self.L, self.H_het, self.H_hom, self.K_het, self.K_hom, self.birth_vec = self._flows_to_transition_matrices(S_mapping, IP_mapping, age_groups, 4)
        self.O, self.R_het, self.R_hom, D = self._flows_to_post_simulation_state_mappings(S_mapping, IP_mapping, 4)

        # --> remaining for user: omega, gamma, beta_monthly, p_hom_monthly, S0, I0, P0
    
        pass

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

        return deaths


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
