import streamlit as st
from pref_voting.voting_methods import * 
from pref_voting.generate_spatial_profiles import *   
from pref_voting.utility_functions import *
from pref_voting.probabilistic_methods import *
from pref_voting.profiles import *
from pref_voting.utility_methods import *
import pandas as pd
import altair as alt
import os.path
import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from utils import load_main_dataframe, load_uncertainty_dataframe

st.set_page_config(
    page_title="Differences in Expected Social Utility Performance",
    page_icon="📊",
)

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

st.title("Differences in Expected Social Utility Performance")

polarized_df, unpolarized_df, all_voting_methods = load_main_dataframe()

print(all_voting_methods)

if 'use_uncertainty' not in st.session_state:
    st.session_state.use_uncertainty = False

if not st.session_state.use_uncertainty:
    st.session_state["all_voting_methods"] = all_voting_methods
    st.session_state["default_vm_list"] = default_vm_list

with st.sidebar:

    vis_type = st.radio('', ["Average over all options",  "Average over all options except..."], key=None, disabled=False, label_visibility="collapsed")
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
    
    # st.subheader("Voting Methods")

    select_all_vms = True #st.checkbox("Show all voting methods")
    # if not select_all_vms:
    #     selected_vms = st.multiselect(
    #         'Select Voting Methods', 
    #         options=st.session_state["all_voting_methods"],
    #         default=st.session_state["default_vm_list"],
    #         disabled=select_all_vms)

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
        elif election_type == "Unpolarized":
            filtered_df = filtered_unpolarized_df
        elif election_type == "Both":
            filtered_df = pd.concat([filtered_polarized_df, filtered_unpolarized_df], ignore_index=True)
        # if not select_all_vms:
        #     filtered_df = filtered_df[filtered_df['vm'].isin(selected_vms)]

    elif not avg_all and (len(avg_types) > 0 or show_all_options):
        if election_type == "Polarized":
            filtered_df = filter_polarized_df(filtered_polarized_df, key_values_for_polarized_elections)
        elif election_type == "Unpolarized":
            filtered_df = filter_unpolarized_df(filtered_unpolarized_df, key_values_for_unpolarized_elections)
        elif election_type == "Both":
            filtered_df = pd.concat([
                filter_polarized_df(filtered_polarized_df, key_values_for_polarized_elections), 
                filter_unpolarized_df(filtered_unpolarized_df, key_values_for_unpolarized_elections)], 
                ignore_index=True)

        # if not select_all_vms:
        #     filtered_df = filtered_df[filtered_df['vm'].isin(selected_vms)]


def calculate_z_p(alpha, m):
    """
    Calculate z_p where p = 1 - alpha/m
    
    Parameters:
    alpha: significance level (e.g., 0.05)
    m: number of comparisons (e.g., 1260)
    
    Returns:
    z_p: the upper p-quantile of the standard normal distribution
    """
    p = 1 - alpha/m
    z_p = stats.norm.ppf(p)
    return z_p


def calculate_mcse(vm1_est_std_errors, vm2_est_std_errors, n):
    return np.sqrt(np.sum(np.square(vm1_est_std_errors + vm2_est_std_errors)) / n**2)

if (not avg_all and len(avg_types) == 0 and not show_all_options) or filtered_df.empty: 
    if ((len(key_values_for_unpolarized_elections['num_dims']) == 1 and len(key_values_for_unpolarized_elections['voter_utility']) == 1) and (key_values_for_unpolarized_elections['num_dims'][0] == 1 and  key_values_for_unpolarized_elections['voter_utility'][0] == 'Matthews')):
        st.error(f'The Matthews utility function is not defined for 1 dimension.', icon="⚠️")
    else:
        st.warning(f'Please select at least one value for each parameter.', icon="⚠️")

else:

    vms = filtered_df['vm'].unique()
    #st.write(f"Number of voting methods: **{len(vms)}**")
    sig_level = 0.05
    num_comparisons = (len(vms) - 1) * (len(vms) - 2) 
    #st.write(f"Number of comparisons: **{num_comparisons}**")
    z_p = calculate_z_p(sig_level, num_comparisons)

    # Create selectbox for vm1
    vm1 = st.selectbox(
        "Select a voting method:",
        sorted(vms),
        key="method1"
    )

    # Remove the first selection from options for the second selectbox
    remaining_vms = sorted([m for m in vms if m != vm1])

    # Initialize session state for select all
    if 'select_all_flag' not in st.session_state:
        st.session_state.select_all_flag = False

    # Add button to select all

    col1, col2 = st.columns([3, 1])

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Select All", type="secondary"):
            st.session_state.method2 = remaining_vms
            st.rerun()

    with col1:
        vm2s = st.multiselect(
            "Select voting methods to compare:",
            remaining_vms,
            key="method2"
        )

    vm1_df = filtered_df[filtered_df['vm'] == vm1]

    if len(vm2s) == 0:
        st.warning(f'Please select at least one voting method to compare against {vm1}.', icon="⚠️")
        st.stop()

    # Collect data
    table_data = {
        "Method 2": [],
        "Difference": [],
        "MCSE": [],
        "Significant": [],
        "Better Method": [],
    }

    for vm2 in vm2s:
        vm2_df = filtered_df[filtered_df['vm'] == vm2]
        diff = np.mean(vm1_df['exp_soc_util_perf'].values) - np.mean(vm2_df['exp_soc_util_perf'].values)
        mcse = calculate_mcse(vm1_df['est_std_error'].values, vm2_df['est_std_error'].values, len(vm1_df))
        is_significant = abs(diff) > z_p * mcse
        
        table_data["Method 2"].append(vm2)
        table_data["Difference"].append(diff)
        table_data["MCSE"].append(mcse)
        # Store as string instead of boolean to avoid checkbox rendering
        table_data["Significant"].append("✔" if is_significant else "✖")
        table_data["Better Method"].append(vm1 if diff > 0 else vm2)

    # Create DataFrame and sort by difference (largest to smallest)
    df_table = pd.DataFrame(table_data)
    df_table = df_table.sort_values('Difference', ascending=False)

    # Style the dataframe
    def style_table(row):
        """Apply color styling based on difference value"""
        diff = row["Difference"]
        is_significant = row["Significant"] == "✔"  # Check against the string now
        
        if diff > 0:
            # Method 1 is better - green tones
            color = '#d4edda' if is_significant else '#f0f8f0'
            text_color = '#155724' if is_significant else '#3d6e3d'
        else:
            # Method 2 is better - red tones  
            color = '#f8d7da' if is_significant else '#fff5f5'
            text_color = '#721c24' if is_significant else '#6e3d3d'
        
        return [f'background-color: {color}; color: {text_color}'] * len(row)

    # Format the numeric columns
    styled_df = df_table.style.apply(style_table, axis=1)
    styled_df = styled_df.format({
        "Difference": "{:+.7f}",
        "MCSE": "{:.7f}",
        # No need to format "Significant" since it's already a string
    })

    # Apply text alignment using set_properties
    styled_df = styled_df.set_properties(**{'text-align': 'center'}, subset=['Difference', 'MCSE', 'Significant'])
    styled_df = styled_df.set_properties(**{'text-align': 'left'}, subset=['Method 2', 'Better Method'])

    # Also set table styles for headers and widths
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},  # Center all headers first
        {'selector': 'th:nth-child(1)', 'props': [('text-align', 'left')]},  # Then left-align first header
        {'selector': 'th:nth-child(5)', 'props': [('text-align', 'left')]},  # And last header
        {'selector': 'td:nth-child(1)', 'props': [('min-width', '150px')]},
        {'selector': 'td:nth-child(2)', 'props': [('width', '120px')]},
        {'selector': 'td:nth-child(3)', 'props': [('width', '120px')]},
        {'selector': 'td:nth-child(4)', 'props': [('width', '100px')]},
        {'selector': 'td:nth-child(5)', 'props': [('min-width', '150px')]},
    ])

    st.write("### Differences in Expected Social Utility Performance")
    st.write(f"**Comparing {vm1} (Method 1) against {len(vm2s)} voting methods**")

    # Statistical information box
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("# Probability Models", f"{len(vm1_df)}")
    with col2:
        st.metric("Comparisons", f"{len(vm2s)} of {len(vms)-1}")
    with col3:
        st.metric("Critical z-value", f"{z_p:.3f}")
    with col4:
        st.metric("Significance Level", f"{sig_level:.0%}")

    st.caption(f"**Difference** = Method 1 ({vm1}) minus Method 2 performance. Positive values favor Method 1.")
    st.write("🟢 **Green** = Method 1 performs better | 🔴 **Red** = Method 2 performs better | **Darker shade** = Statistically significant")

    # Display with increased height
    st.dataframe(styled_df, hide_index=True, use_container_width=True, height=500)

    # Optional: Summary statistics
    if len(df_table) > 0:
        sig_count = (df_table['Significant'] == "✔").sum()
        vm1_wins = ((df_table['Difference'] > 0) & (df_table['Significant'] == "✔")).sum()
        vm2_wins = ((df_table['Difference'] < 0) & (df_table['Significant'] == "✔")).sum()
        
        st.write("---")
        st.write("**Summary:**")
        st.write(f"• Significant differences: **{sig_count}/{len(df_table)}**")
        st.write(f"• {vm1} significantly better: **{vm1_wins}**")
        st.write(f"• Others significantly better: **{vm2_wins}**")