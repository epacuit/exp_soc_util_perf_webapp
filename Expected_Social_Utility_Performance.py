import streamlit as st
from pref_voting.voting_methods import * 
from pref_voting.generate_spatial_profiles import *   
from pref_voting.utility_functions import *
from pref_voting.probabilistic_methods import *
from pref_voting.grade_methods import *
import pandas as pd
import altair as alt
import os.path
import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal
import pandas as pd

from utils import load_main_dataframe, load_uncertainty_dataframe, load_condorcet_efficiency_data

st.set_page_config(
    page_title="Expected Social Utility Performance of Voting Methods",
    page_icon="📊",
)

condorcet_vms = [
    knockout,
    condorcet_plurality,
    condorcet,
    copeland,
    copeland_local_borda,
    copeland_global_borda,
    benham, 
    bottom_two_runoff_instant_runoff,
    baldwin,
    weak_nanson,
    raynaud,
    minimax,
    stable_voting,
    beat_path,
    ranked_pairs_zt,
    split_cycle,
    daunou,
    blacks,
    condorcet_irv,
    smith_irv, 
    woodall, 
    river_zt,
    smith_minimax, 
    tideman_alternative_smith,
    smith_set,
    copeland_global_minimax,
    loss_trimmer,
    superior_voting,
] 
condorcet_vm_names = [vm.name for vm in condorcet_vms ] + ['Bottom-Two-Runoff IRV', 'Tideman Alternative Smith',"Black's"]


prob_vms = []
grade_vms = [approval]

default_vm_list = sorted([
    "Plurality", 
    "Condorcet Plurality",
    "Instant Runoff",
    "Condorcet IRV",
    "Borda", 
    "Approval", 
    "Copeland-Global-Borda", 
    "Copeland-Global-Minimax", 
    "Plurality with Runoff", 
    "Minimax", 
    "Stable Voting",
    "Beat Path", 
    "Ranked Pairs ZT", 
    "Split Cycle",
    "Black's",
    ])
def select_normalizations():
    all_normalizations = ["none", "range"]
    normalizations_range = st.multiselect(
        'Normalization(s)',
        options=all_normalizations,
        default=all_normalizations[0:1]
    )
    return normalizations_range

def select_num_cands():
    all_num_cands = [3, 4, 5, 6, 7, 8, 9, 10]

    st.subheader('Number(s) of candidates')
    num_cands_range = st.slider('', min_value=min(all_num_cands), max_value=max(all_num_cands), value=(min(all_num_cands), min(all_num_cands)), label_visibility="collapsed")
    
    num_cands_range = list(range(num_cands_range[0], num_cands_range[1] + 1))
    return num_cands_range

def select_num_voters():
    st.subheader("Number(s) of voters")
    all_num_voters = [11, 101, 1001]
    num_voters_range = [v for v in all_num_voters if st.checkbox(f"{v}", key=f"num_voters{v}", value=(v==11))]
    return num_voters_range

def select_num_dims():
    st.subheader("Number(s) of dimensions")
    all_num_dims = [1, 2, 4, 8]
    num_dims_range = [d for d in all_num_dims if st.checkbox(f"{d}", key=f"dims{d}", value=(d==2))]
    return num_dims_range

def select_dispersion():
    st.subheader("Relative dispersion(s)")
    all_dispersion = [0.5, 1]
    rel_dispersion_range = [d for d in all_dispersion if st.checkbox(f"{d}", key=f"dispersion{d}", value=(d==0.5))]
    return rel_dispersion_range

def select_correlation():
    st.subheader("Correlation(s)")
    all_correlation = [0, 0.5]
    correlation_range = [c for c in all_correlation if st.checkbox(f"{c}", key=f"correlations{c}", value=(c==0))]
    return correlation_range

def select_voter_utilities():
    all_voter_utilities = {
        "Linear": linear_utility, 
        "Quadratic": quadratic_utility, 
        "Shepsle": shepsle_utility,
        "Matthews": matthews_utility,
        "Mixed Proximity-RM": mixed_rm_utility,
        "RM": rm_utility    
    }
    st.subheader('Voter utility function(s)')
    voter_utility_range = st.multiselect(
        '',
        options=list(all_voter_utilities.keys()),
        default=list(all_voter_utilities.keys())[0:1], 
        label_visibility="collapsed"
    )
    return voter_utility_range

def select_num_dims_polarized(election_type):
    all_num_dims_polarized = ["one", "half", "all"]
    st.write("Number(s) of polarized dimensions")
    num_dims_polarized_range = [d for d in all_num_dims_polarized if st.checkbox(f"{d}", key=f"num_dims_polarized{d}", value=(d=="one"), disabled = election_type == "Unpolarized")]
    return num_dims_polarized_range

def select_subpopulation_stds(election_type):
    st.subheader("Subpopulation standard deviation(s)")
    all_subpopulation_stds = [1.0, 0.5]
    subpopulation_stds_range = [s for s in all_subpopulation_stds if st.checkbox(f"{s}", key=f"subpopulation_stds{s}", value=(s==1.0), disabled = election_type == "Unpolarized")]
    return subpopulation_stds_range

def select_num_centrist_cands(election_type):
    st.subheader("Number(s) of centrist candidate")
    all_num_centrist_cands = ["none", "half", "all"]
    num_centrist_cands_range = [c for c in all_num_centrist_cands if st.checkbox(f"{c}", key=f"num_centrist_cands{c}", value=(c=="none"), disabled = election_type == "Unpolarized")]
    return num_centrist_cands_range

def select_prob_centrist_voters(election_type):
    st.subheader("Probability(s) of centrist voter")
    all_prob_centrist_voters = [0.0, 0.5, 1.0]
    prob_centrist_voters_range = [p for p in all_prob_centrist_voters if st.checkbox(f"{p}", key=f"prob_centrist_voters{p}", value=(p==0.0), disabled = election_type == "Unpolarized")]
    return prob_centrist_voters_range

def select_election_type():
    election_type = st.radio(
            f"Election types",
            ["Unpolarized", "Polarized", "Both"],
            captions=[
                "Restrict to unpolarized elections.",
                "Restrict to polarized elections.",
                "Include both polarized and unpolarized elections.",
            ],
        ) 
    return election_type

def select_with_uncertainty():
    return st.radio(
            f"Voter Uncertainty",
            ["No Uncertainty", "Uncertainty", "Both"],
            captions=[
                "Restrict to elections without uncertainty about candidate positions.",
                "Restrict to elections with uncertainty about candidate positions.",
                "Include elections both with and without uncertainty about candidate positions.",
            ],
        ) 

def select_use_perceived_uncertainty():
    st.subheader("Use perceived uncertainty")
    all_use_perceived_uncertainty = [True, False]
    use_perceived_uncertainty_range = [s for s in all_use_perceived_uncertainty if st.checkbox(f"{s}", key=f"use_perceived_uncertainty{s}", value=(s==1.0))]
    return use_perceived_uncertainty_range


@st.cache_resource(show_spinner="Filtering data...")
def filter_unpolarized_df(df, key_columns_values):
    mask = pd.Series(True, index=df.index)
    for column, values in key_columns_values.items():
        if pd.isna(values).any():
            mask &= df[column].isin(values) | df[column].isna()
        else:
            mask &= df[column].isin(values)
    return df[mask]

@st.cache_resource(show_spinner="Filtering data...")
def filter_polarized_df(df, key_columns_values):

    mask = pd.Series(True, index=df.index)
    for column, values in key_columns_values.items():
        if column != "num_dims_polarized" and column != "num_centrist_cands" and column != "prob_centrist_voters" and column != "polarization_distance":
            if pd.isna(values).any():
                mask &= df[column].isin(values) | df[column].isna()
            else:
                mask &= df[column].isin(values)
    filtered_df = df[mask]

    # Convert num_dims_polarized_range to their numeric equivalents based on num_dims
    polarized_mapping = {"one": 1, "half": lambda x: x // 2, "all": lambda x: x}
    num_dims_polarized_numeric = {nd: [polarized_mapping[p](nd) if callable(polarized_mapping[p]) else polarized_mapping[p] for p in key_columns_values['num_dims_polarized'] if p!= 0] for nd in key_columns_values['num_dims']}

    # Convert num_centrist_cands_range to their numeric equivalents based on num_cands
    centrist_mapping = {"none": 0, "half": lambda x: x // 2, "all": lambda x: x}
    num_centrist_cands_numeric = {nc: [centrist_mapping[c](nc) if callable(centrist_mapping[c]) else centrist_mapping[c] for c in key_columns_values['num_centrist_cands']] for nc in key_columns_values['num_cands']}

    # Filtering based on num_dims and num_dims_polarized pairs
    dims_mask = pd.Series(False, index=df.index)
    for nd in key_columns_values['num_dims']:
        for nd_p in num_dims_polarized_numeric[nd]:
            if nd_p != 0:
                dims_mask |= (df['num_dims'] == nd) & (df['num_dims_polarized'] == nd_p)

    cands_mask = pd.Series(False, index=df.index)
    for nc in key_columns_values['num_cands']:
        for nc_c in num_centrist_cands_numeric[nc]:
            if not np.isnan(nc_c):
                cands_mask |= (df['num_cands'] == nc) & (df['num_centrist_cands'] == nc_c)

    # Filtering based on num_cands and num_centrist_cands pairs
    prob_centrist_voters_mask = pd.Series(False, index=df.index)
    for prob in key_columns_values['prob_centrist_voters']: 
        prob_centrist_voters_mask |=  (df['prob_centrist_voters'] == prob)

    # Combine masks and apply to dataframe
    combined_mask = dims_mask & cands_mask & prob_centrist_voters_mask
    filtered_polarized_df = filtered_df[combined_mask]
    return filtered_polarized_df


key_values_for_unpolarized_elections = {
    'num_cands':[3, 4, 5, 6, 7, 8, 9, 10], 
    'num_voters':[11, 101, 1001], 
    'num_dims': [1, 2, 4, 8], 
    'correlation': [0, 0.5], 
    'rel_dispersion':[0.5, 1], 
    'voter_utility':[
        "Linear", 
        "Quadratic", 
        "Shepsle", 
        "Matthews", 
        "Mixed Proximity-RM", 
        "RM"], 
    'num_dims_polarized': [0], 
    'subpopulation_std': [1.0], 
    'polarization_distance': [np.nan], 
    'num_centrist_cands': [np.nan], 
    'prob_centrist_voters': [np.nan],
}

key_values_for_polarized_elections = {
    'num_cands':[3, 4, 5, 6, 7, 8, 9, 10], 
    'num_voters':[11, 101, 1001], 
    'num_dims': [1, 2, 4, 8], 
    'correlation': [0, 0.5], 
    'rel_dispersion':[0.5, 1], 
    'voter_utility':[
        "Linear", 
        "Quadratic", 
        "Shepsle", 
        "Matthews", 
        "Mixed Proximity-RM",
        "RM"], 
    'num_dims_polarized': ["one", "half", "all"], 
    'subpopulation_std': [1.0, 0.5], 
    'polarization_distance': [1], 
    'num_centrist_cands': ["none", "half", "all"], 
    'prob_centrist_voters': [0.0, 0.5, 1.0],
}

st.title("Expected Social Utility Performance of Voting Methods")

polarized_df, unpolarized_df, all_voting_methods = load_main_dataframe()

condorcet_eff_df = load_condorcet_efficiency_data("./data/condorcet_efficiency_data.csv.zip")

print(all_voting_methods)

if 'use_uncertainty' not in st.session_state:
    st.session_state.use_uncertainty = False

if not st.session_state.use_uncertainty:
    st.session_state["all_voting_methods"] = all_voting_methods
    st.session_state["default_vm_list"] = default_vm_list

with st.sidebar:

    vis_type = st.radio('', ["Show all options", "Average over all options",  "Average over all options except..."], key=None, disabled=False, label_visibility="collapsed")
    avg_all = vis_type == "Average over all options"
    show_all_options = vis_type == "Show all options"
    if vis_type == "Average over all options except...":
        avg_types = st.multiselect("", [
            "Number of Candidates",
            "Number of Voters",
            "Number of Dimensions",
            "Relative Dispersion",
            "Correlation",
            "Voter Utilities",
            "Number of Polarized Dimensions",
            "Subpopulation Standard Deviation(s)",
            "Number of Centrist Candidates",
            "Probability of Centrist Voters",
        ]  + ([] if not st.session_state.use_uncertainty else ["Use Perceived Uncertainty"]),  
        key=None, 
        help=None, 
        placeholder="Choose an option", 
        label_visibility="collapsed")
    else: 
        avg_types = []

    if avg_all or len(avg_types) > 0 or show_all_options:
        election_type = select_election_type()
    
    if (not avg_all and "Number of Candidates" in avg_types) or show_all_options:
        num_cands_range = select_num_cands()
        key_values_for_unpolarized_elections['num_cands'] = num_cands_range 
        key_values_for_polarized_elections['num_cands'] = num_cands_range 

    if (not avg_all and "Number of Voters" in avg_types) or show_all_options:
        num_voters_range = select_num_voters()
        key_values_for_unpolarized_elections['num_voters'] = num_voters_range 
        key_values_for_polarized_elections['num_voters'] = num_voters_range 

    if (not avg_all and "Number of Dimensions" in avg_types) or show_all_options:
        num_dims_range = select_num_dims()
        key_values_for_unpolarized_elections['num_dims'] = num_dims_range
        key_values_for_polarized_elections['num_dims'] = num_dims_range


    if (not avg_all and "Correlation" in avg_types) or show_all_options:
        correlation_range = select_correlation()
        key_values_for_unpolarized_elections['correlation'] = correlation_range
        key_values_for_polarized_elections['correlation'] = correlation_range

    if (not avg_all and "Relative Dispersion" in avg_types) or show_all_options:
        rel_dispersion_range = select_dispersion()
        key_values_for_unpolarized_elections['rel_dispersion'] = rel_dispersion_range
        key_values_for_polarized_elections['rel_dispersion'] = rel_dispersion_range

    if (not avg_all and "Voter Utilities" in avg_types) or show_all_options:
        voter_utility_range = select_voter_utilities()
        key_values_for_unpolarized_elections['voter_utility'] = voter_utility_range
        key_values_for_polarized_elections['voter_utility'] = voter_utility_range

    if (not avg_all and "Number of Polarized Dimensions" in avg_types) or show_all_options:
        num_dims_polarized_range =  select_num_dims_polarized(election_type)
        key_values_for_polarized_elections['num_dims_polarized'] = num_dims_polarized_range

    if (not avg_all and "Subpopulation Standard Deviation(s)" in avg_types) or show_all_options:
        subpopulation_stds_range =  select_subpopulation_stds(election_type)
        key_values_for_polarized_elections['subpopulation_std'] = subpopulation_stds_range

    if (not avg_all and "Number of Centrist Candidates" in avg_types) or show_all_options:
        num_centrist_cands_range =  select_num_centrist_cands(election_type)
        key_values_for_polarized_elections['num_centrist_cands'] = num_centrist_cands_range

    if (not avg_all and "Probability of Centrist Voters" in avg_types) or show_all_options:
        prob_centrist_voters_range =  select_prob_centrist_voters(election_type)
        key_values_for_polarized_elections['prob_centrist_voters'] = prob_centrist_voters_range
    
    st.subheader("Voting Methods")

    select_all_vms = st.checkbox("Show all voting methods")
    if not select_all_vms:
        selected_vms = st.multiselect(
            'Select Voting Methods', 
            options=st.session_state["all_voting_methods"],
            default=st.session_state["default_vm_list"],
            disabled=select_all_vms)

    st.subheader("Modifications")

    with_uncertainty = select_with_uncertainty()

    no_uncertainty = with_uncertainty == "No Uncertainty"

    only_uncertainty = with_uncertainty == "Uncertainty"

    uncertainty_plus_no_uncertainty = with_uncertainty == "Both"

    if only_uncertainty or uncertainty_plus_no_uncertainty: 
        if (not avg_all and "Use Perceived Uncertainty" in avg_types) or show_all_options:
            use_perceived_uncertainty_range =  select_use_perceived_uncertainty()    
        else:  
            use_perceived_uncertainty_range = [True, False]

    if only_uncertainty or uncertainty_plus_no_uncertainty: 
        polarized_uncertainty_df, unpolarized_uncertainty_df, all_voting_methods_uncertainty = load_uncertainty_dataframe()
        if not st.session_state.use_uncertainty:
            st.session_state.use_uncertainty = True
            st.session_state["all_voting_methods"] = all_voting_methods_uncertainty
            st.session_state["default_vm_list"] = [_vm for _vm in default_vm_list if _vm in all_voting_methods_uncertainty]
            st.rerun()
    else: 
        if st.session_state.use_uncertainty: 
            st.session_state.use_uncertainty = False
            st.rerun()

    normalizations_range = select_normalizations()

    if no_uncertainty:
        filtered_polarized_df = polarized_df[polarized_df["normalization"].isin(normalizations_range)]

        filtered_unpolarized_df = unpolarized_df[unpolarized_df["normalization"].isin(normalizations_range)]

        if 'use_perceived_uncertainty' in key_values_for_unpolarized_elections:
            del key_values_for_unpolarized_elections['use_perceived_uncertainty']
        if 'use_perceived_uncertainty' in key_values_for_polarized_elections:
            del key_values_for_polarized_elections['use_perceived_uncertainty']

        restricted_condorcet_eff_df = condorcet_eff_df[condorcet_eff_df["normalization"].isin(normalizations_range)]

        filtered_polarized_condorcet_eff_df = restricted_condorcet_eff_df[restricted_condorcet_eff_df['num_dims_polarized'] != 0]

        filtered_unpolarized_condorcet_eff_df = restricted_condorcet_eff_df[restricted_condorcet_eff_df['num_dims_polarized'] == 0]

    elif only_uncertainty:
        filtered_polarized_df = polarized_uncertainty_df[polarized_uncertainty_df["normalization"].isin(normalizations_range)]

        filtered_unpolarized_df = unpolarized_uncertainty_df[unpolarized_uncertainty_df["normalization"].isin(normalizations_range)]

        key_values_for_unpolarized_elections['use_perceived_uncertainty'] = use_perceived_uncertainty_range
        key_values_for_polarized_elections['use_perceived_uncertainty'] = use_perceived_uncertainty_range

    elif uncertainty_plus_no_uncertainty:

        filtered_polarized_df_1 = polarized_df[polarized_df["normalization"].isin(normalizations_range)]

        filtered_unpolarized_df_1 = unpolarized_df[unpolarized_df["normalization"].isin(normalizations_range)]

        filtered_polarized_df_2 = polarized_uncertainty_df[polarized_uncertainty_df["normalization"].isin(normalizations_range)]

        filtered_unpolarized_df_2 = unpolarized_uncertainty_df[unpolarized_uncertainty_df["normalization"].isin(normalizations_range)]

        key_values_for_unpolarized_elections['use_perceived_uncertainty'] = use_perceived_uncertainty_range + ["None"]
        key_values_for_polarized_elections['use_perceived_uncertainty'] = use_perceived_uncertainty_range + ["None"]

        filtered_polarized_df_1['use_perceived_uncertainty'] = "None"
        filtered_unpolarized_df_1['use_perceived_uncertainty'] = "None"

        filtered_polarized_df_1 = filtered_polarized_df_1[filtered_polarized_df_1['vm'].isin(filtered_polarized_df_2['vm'].unique())]
        filtered_unpolarized_df_1 = filtered_unpolarized_df_1[filtered_unpolarized_df_1['vm'].isin(filtered_unpolarized_df_2['vm'].unique())]

        filtered_polarized_df = pd.concat([filtered_polarized_df_1, filtered_polarized_df_2], ignore_index=True)

        filtered_unpolarized_df = pd.concat([filtered_unpolarized_df_1, filtered_unpolarized_df_2], ignore_index=True)

    if avg_all: 
        if election_type == "Polarized":
            filtered_df = filtered_polarized_df
            if no_uncertainty:
                filtered_condorcet_eff_df = filtered_polarized_condorcet_eff_df
        elif election_type == "Unpolarized":
            filtered_df = filtered_unpolarized_df
            if no_uncertainty:
                filtered_condorcet_eff_df = filtered_unpolarized_condorcet_eff_df
        elif election_type == "Both":
            filtered_df = pd.concat([filtered_polarized_df, filtered_unpolarized_df], ignore_index=True)
            if no_uncertainty:
                filtered_condorcet_eff_df = pd.concat([filtered_polarized_condorcet_eff_df,  filtered_unpolarized_condorcet_eff_df], ignore_index=True)
        if not select_all_vms:
            filtered_df = filtered_df[filtered_df['vm'].isin(selected_vms)]

    elif not avg_all and (len(avg_types) > 0 or show_all_options):
        if election_type == "Polarized":
            filtered_df = filter_polarized_df(filtered_polarized_df, key_values_for_polarized_elections)
            if no_uncertainty:
                filtered_condorcet_eff_df = filter_polarized_df(filtered_polarized_condorcet_eff_df,key_values_for_polarized_elections) 
        elif election_type == "Unpolarized":
            filtered_df = filter_unpolarized_df(filtered_unpolarized_df, key_values_for_unpolarized_elections)
            if no_uncertainty:
                filtered_condorcet_eff_df = filter_unpolarized_df(filtered_unpolarized_condorcet_eff_df, key_values_for_unpolarized_elections)
        elif election_type == "Both":
            filtered_df = pd.concat([
                filter_polarized_df(filtered_polarized_df, key_values_for_polarized_elections), 
                filter_unpolarized_df(filtered_unpolarized_df, key_values_for_unpolarized_elections)], 
                ignore_index=True)
            if no_uncertainty:
                filtered_condorcet_eff_df = pd.concat([
                    filter_polarized_df(filtered_polarized_condorcet_eff_df,key_values_for_polarized_elections),filter_unpolarized_df(filtered_unpolarized_condorcet_eff_df, key_values_for_unpolarized_elections)],
                    ignore_index=True)

        if not select_all_vms:
            filtered_df = filtered_df[filtered_df['vm'].isin(selected_vms)]

if (not avg_all and len(avg_types) == 0 and not show_all_options) or  filtered_df.empty: 
    if ((len(key_values_for_unpolarized_elections['num_dims']) == 1 and len(key_values_for_unpolarized_elections['voter_utility']) == 1) and (key_values_for_unpolarized_elections['num_dims'][0] == 1 and  key_values_for_unpolarized_elections['voter_utility'][0] == 'Matthews')):
        st.error(f'The Matthews utility function is not defined for 1 dimension.', icon="⚠️")
    else:
        st.warning(f'Please select at least one value for each parameter.', icon="⚠️")

else: 
    num_simulations = int(len(filtered_df) / len(filtered_df['vm'].unique()))
    if num_simulations > 1:
        tab1, tab2 = st.tabs(["Average Expected Social Utility Performance", "Data"])
    else:
        tab1, tab2 = st.tabs(["Expected Social Utility Performance", "Data"])
    with tab1: 
        col1, col2 = st.columns(2)
        with col1: 
            vm_order = st.radio(
                'How should the voting methods be ordered?',
                ('by value', 'alphabetically'))
        with col2:
            show_condorcet = st.checkbox("Show Condorcet Methods")
            if no_uncertainty: 
                show_cond_eff_stats = st.checkbox("Show Condorcet Winner Statistics")
                show_absolute_util_stats = st.checkbox("Show Absolute Utility Statistics")
        graph_height = len(filtered_df['vm'].unique()) * 30 if len(filtered_df['vm'].unique()) * 30 > 350 else 350

        if no_uncertainty and show_cond_eff_stats:
            cond_eff_stats_str = "**Condorcet Winner Statistics**\n\n\n" 
            cond_eff_stats_str += f"Percent of elections with a Condorcet winner: **{round(filtered_condorcet_eff_df['perc_condorcet_winner'].mean() * 100, 3) }%**.\n\n\n" 
            cond_eff_stats_str += f"Among elections with a Condorcet winner, percent in which the Condorcet winner has the highest social utility of any candidate: **{round(filtered_condorcet_eff_df['cond_eff'].mean() * 100, 3)}%**.\n\n\n"
            cond_eff_stats_str += f"The average social utility performance of the Condorcet winner, when it exists: **{round(filtered_condorcet_eff_df['cond_eff'].mean(), 3)}**.\n\n\n"
            st.info(cond_eff_stats_str)

        if no_uncertainty and show_absolute_util_stats:
            abs_util_stats_str = "**Absolute Utility Statistics**\n\n\n" 
            abs_util_stats_str += f"Expected maximum social utility of a candidate: **{round(filtered_condorcet_eff_df['mean_max_util_of_cand'].mean(), 3)}**.\n\n\n" 
            abs_util_stats_str += f"Expected average social utility of all candidates: **{round(filtered_condorcet_eff_df['mean_avg_util_of_cand'].mean(), 3)}**.\n\n\n" 
            abs_util_stats_str += f"Expected minimum social utility of a candidate: **{round(filtered_condorcet_eff_df['mean_min_util_of_cand'].mean(), 3)}**.\n\n\n" 
            st.info(abs_util_stats_str)

        if num_simulations > 1: 

            if only_uncertainty or uncertainty_plus_no_uncertainty:
                graph_title = "Expected Social Utility Performance of Voting Methods, Averaging over Parameters, with Uncertainty"
            else:
                graph_title = "Expected Social Utility Performance of Voting Methods, Averaging over Parameters"
            df_grouped = filtered_df.groupby('vm')['exp_soc_util_perf'].mean().reset_index()

            min_x_val = min(df_grouped['exp_soc_util_perf'].min(), 0)
            bars_avg = (
                alt.Chart(data=df_grouped)
                    .mark_bar()
                        .encode(
                            alt.Y('vm:N', sort=alt.EncodingSortField(field="vm" if vm_order == "alphabetically" else "exp_soc_util_perf", op="sum", order='ascending'  if vm_order == "alphabetically" else 'descending')).title(''),
                            alt.X('exp_soc_util_perf:Q').title(' ').scale(alt.Scale(domain=[min_x_val, 1])),
                            color=alt.Color('exp_soc_util_perf:Q', scale=alt.Scale(scheme='redyellowgreen', domain=[0.3, 1]), legend=None),  # Color based on x-values

                            tooltip=[alt.Tooltip('vm:N', title="Voting Method"),
                            alt.Tooltip('exp_soc_util_perf:Q', format=".4f", title="Overall Average")],
                        )
                        .properties(
                            title=graph_title,
                            height=graph_height,
                            width=900
                        )
                        )
            
            if show_condorcet:
                text = alt.Chart(data=df_grouped).mark_text(
                    align='left',
                    baseline='middle',
                    dx=3,  # Nudges text to right so it doesn't appear on top of the bar
                    color='white'
                ).encode(
                    alt.Y('vm:N', sort=alt.EncodingSortField(field="vm" if vm_order == "alphabetically" else "exp_soc_util_perf", op="sum", order='ascending'  if vm_order == "alphabetically" else 'descending')).title(''),
                    x=alt.value(0),
                    text=alt.condition(
                        alt.FieldOneOfPredicate(field='vm', oneOf=condorcet_vm_names),  # Check if 'vm' is in the predefined list
                        alt.value('Condorcet'),  # display 'Condorcet' if condition is true
                        alt.value('')  # do not display text if condition is false
                    )
                )
                #combined_chart = alt.layer(bars_avg, text)
                combined_chart = (bars_avg + text).configure_axis(
                        labelLimit=500  # Increase label limit to allow wider labels
                    ).configure_axisY(  # Apply configuration to Y-axis
                        labelFontSize=16  # Set label font size
                    )


                st.altair_chart(combined_chart ,  use_container_width=False)
            else: 
                combined_chart = (bars_avg).configure_axis(
                        labelLimit=500  # Increase label limit to allow wider labels
                    ).configure_axisY(  # Apply configuration to Y-axis
                        labelFontSize=16  # Set label font size
                    )

                st.altair_chart(combined_chart,  use_container_width=False)

        else: 
            if only_uncertainty or uncertainty_plus_no_uncertainty:
                graph_title = "Expected Social Utility Performance of Voting Methods, with Uncertainty"
            else:
                graph_title = "Expected Social Utility Performance of Voting Methods"
            filtered_df['lower'] = filtered_df['exp_soc_util_perf'] - filtered_df['est_std_error']
            filtered_df['upper'] = filtered_df['exp_soc_util_perf'] + filtered_df['est_std_error']

            max_exp_soc_util_perf = filtered_df['lower'].max()

            filtered_df['has_max_exp_soc_util_perf'] = filtered_df['upper']  > max_exp_soc_util_perf


            filtered_df['vm'] = filtered_df['vm'].replace('PluralityWRunoff PUT', 'Plurality with Runoff')
            filtered_df['vm'] = filtered_df['vm'].replace('Bottom-Two-Runoff Instant Runoff', 'Bottom-Two-Runoff IRV')

            filtered_df['vm'] = filtered_df['vm'].replace('Tideman Alternative Top Cycle', 'Tideman Alternative Smith')

            updated_condorcet_names = ['Bottom-Two-Runoff IRV', 'Tideman Alternative Smith']

            # Define the sorting condition
            sort_field = "vm" if vm_order == "alphabetically" else "exp_soc_util_perf"
            sort_order = 'ascending' if vm_order == "alphabetically" else 'descending'

            # Create the error bars
            error_bars = alt.Chart(filtered_df).mark_errorbar().encode(
                    alt.Y('vm:N',
                        sort=alt.EncodingSortField(field=sort_field, op="sum", order=sort_order),
                        title=''),
                x='lower:Q',
                x2='upper:Q'
            )
            min_x_val = min(filtered_df['exp_soc_util_perf'].min(), 0)

            # Create the bars
            bars = alt.Chart(filtered_df).mark_bar().encode(
                alt.Y('vm:N',
                    sort=alt.EncodingSortField(field=sort_field, op="sum", order=sort_order),
                    title=''),
                alt.X('exp_soc_util_perf:Q', title=' ', scale=alt.Scale(domain=[min_x_val, 1])),
                tooltip=[alt.Tooltip('vm:N', title="Voting Method"),
                        alt.Tooltip('exp_soc_util_perf:Q', format=".4f", title="Expected Social Utility Performance"),
                        alt.Tooltip('lower:Q', format=".4f", title="Lower Bound"),
                        alt.Tooltip('upper:Q', format=".4f", title="Upper Bound")],
                color=alt.condition(alt.datum.has_max_exp_soc_util_perf, alt.value("orange"), alt.value("steelblue"))
            ).properties(
                title=graph_title,
                height=graph_height,
                width=900
            )

            # Combine charts
            combined_chart = alt.layer(bars, error_bars)
            if show_condorcet:
                text = alt.Chart(data=filtered_df).mark_text(
                    align='left',
                    baseline='middle',
                    dx=3,  # Nudges text to right so it doesn't appear on top of the bar
                    color='white'
                ).encode(
                    alt.Y('vm:N',
                        sort=alt.EncodingSortField(field=sort_field, op="sum", order=sort_order),
                        title=''),
                    x=alt.value(0),
                    text=alt.condition(
                        alt.FieldOneOfPredicate(field='vm', oneOf=condorcet_vm_names),  # Check if 'vm' is in the predefined list
                        alt.value('Condorcet'),  # display 'Condorcet' if condition is true
                        alt.value('')  # do not display text if condition is false
                    )
                )

                combined_chart = (bars  + text).configure_axis(
                        labelLimit=500  # Increase label limit to allow wider labels
                    ).configure_axisY(  # Apply configuration to Y-axis
                        labelFontSize=16  # Set label font size
                    )

                st.altair_chart(combined_chart,  use_container_width=False)
            else: 
                combined_chart = (bars).configure_axis(
                        labelLimit=500  # Increase label limit to allow wider labels
                    ).configure_axisY(  # Apply configuration to Y-axis
                        labelFontSize=16  # Set label font size
                    )
                st.altair_chart(combined_chart,  use_container_width=False)

            st.write(f"A bar is orange when the expected social utility performance plus the half width (the upper bound) is greater than **{round(max_exp_soc_util_perf, 4)}**")


    with tab2:
        if num_simulations > 1:
            st.write(f"{num_simulations} simulations selected")
        else:
            st.write(f"{num_simulations} simulation selected")
        df_size_mb = filtered_df.memory_usage(deep=True).sum() / (1024 ** 2)
        if df_size_mb < 200:
            # drop Unnamed: 0.1 column if it exists
            if "Unnamed: 0.1" in filtered_df.columns:
                filtered_df.drop("Unnamed: 0.1", axis=1, inplace=True)
            if only_uncertainty or uncertainty_plus_no_uncertainty:
                st.subheader("Data for Expected Social Utility Performance (with Uncertainty)")
            else: 
                st.subheader("Data for Expected Social Utility Performance")

            st.write(filtered_df)

        else:
            st.warning(f"The filtered dataframe for the Expected Social Utility Performance is too large to display ({df_size_mb:.2f} MB).")

        if no_uncertainty:
            df_cond_eff_size_mb = filtered_condorcet_eff_df.memory_usage(deep=True).sum() / (1024 ** 2)
            if df_cond_eff_size_mb < 200:

                st.subheader("Data for Condorcet Winner and Absolute Utility Statistics")
                st.write(filtered_condorcet_eff_df)
            else:
                st.warning(f"The filtered dataframe for the Condorcet winner and absolute utility statistics is too large to display ({df_size_mb:.2f} MB).")

