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
    page_title="Average Difference of Expected Social Utility Performance",
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


st.title("Average Difference of Expected Social Utility Performance")

polarized_df, unpolarized_df, all_voting_methods = load_main_dataframe()

if 'use_uncertainty' not in st.session_state:
    st.session_state.use_uncertainty = False

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


### Main App

if (not avg_all and len(avg_types) == 0 and not show_all_options) or  filtered_df.empty: 
    if ((len(key_values_for_unpolarized_elections['num_dims']) == 1 and len(key_values_for_unpolarized_elections['voter_utility']) == 1) and (key_values_for_unpolarized_elections['num_dims'][0] == 1 and  key_values_for_unpolarized_elections['voter_utility'][0] == 'Matthews')):
        st.error(f'The Matthews utility function is not defined for 1 dimension.', icon="⚠️")
    else:
        st.warning(f'Please select at least one value for each parameter.', icon="⚠️")

else:
    vms = filtered_df['vm'].unique()
    if st.session_state.use_uncertainty:
        default_vms_to_compare = [
            "Plurality", 
            "Borda", 
            "Instant Runoff", 
            "Black's", 
            "Condorcet IRV", 
            "Approval", 
            "Stable Voting"]
    else: 
        default_vms_to_compare = [
            "Plurality", 
            "Borda", 
            "Instant Runoff", 
            "Condorcet Plurality", 
            "Black's", 
            "Condorcet IRV"]
    vms_to_compare = st.multiselect(
        "Select up to 7 voting methods", 
        options=sorted(vms),
        default = sorted(default_vms_to_compare),
        max_selections=7)
        
    if len(vms_to_compare) < 3: 
        st.warning("Please select 3 or more voting methods.")
    else:
        with st.spinner('Generating heatmap...'):

            filtered_df2 = filtered_df[filtered_df['vm'].isin(vms_to_compare)]    
            # Initialize a matrix to store the average differences and another for confidence intervals
            avg_diff_matrix = np.zeros((len(vms_to_compare), len(vms_to_compare)))
            ci_matrix = np.zeros((len(vms_to_compare), len(vms_to_compare)))

            # Calculate the average difference and confidence intervals between pairs of voting methods
            for i, vm1 in enumerate(vms_to_compare):
                for j, vm2 in enumerate(vms_to_compare):
                    perf1 = filtered_df2[filtered_df2['vm'] == vm1]['exp_soc_util_perf']
                    perf2 = filtered_df2[filtered_df2['vm'] == vm2]['exp_soc_util_perf']

                    differences = perf1.values - perf2.values
                    avg_diff = np.mean(differences)
                    avg_diff_matrix[i, j] = avg_diff

                    # Calculate the confidence interval of the average difference
                    ci = stats.t.interval(0.95, len(differences)-1, loc=avg_diff, scale=stats.sem(differences))
                    ci_range = (ci[1] - ci[0]) / 2  # half the width of the confidence interval
                    ci_matrix[i, j] = ci_range

            # Fill diagonal with zeros
            np.fill_diagonal(avg_diff_matrix, 0)
            np.fill_diagonal(ci_matrix, 0)

            # Create a mask for the upper triangle to include the diagonal
            mask = np.triu(np.ones_like(avg_diff_matrix, dtype=bool), k=1)

            plt.figure(figsize=(14, 12))
            cmap = sns.diverging_palette(10, 133, as_cmap=True)  # Dark green to dark red
            ax = sns.heatmap(avg_diff_matrix, annot=False, fmt=".3f",
                            mask=mask, cmap=cmap, 
                            cbar_kws={'label': ''},
                            annot_kws={'fontsize':16})

            # Adjust the colorbar tick labels' fontsize
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=16)

            # Adjust the text color based on the background color
            for i in range(len(vms_to_compare)):
                for j in range(len(vms_to_compare)):
                    if j <= i:
                        ci_text = f"±{ci_matrix[i, j]:.3f}"
                        # Get the color of the cell
                        color = ax.collections[0].cmap(ax.collections[0].norm(avg_diff_matrix[i, j]))
                        text_color = 'white' if sum(color[:3]) / 3 < 0.5 else 'black'
                        ax.text(j + 0.5, i + 0.5, f"{avg_diff_matrix[i, j]:.3f}", color=text_color, ha='center', fontsize=16)
                        ax.text(j + 0.5, i + 0.75, ci_text, color=text_color, ha='center', fontsize=16)

            plt.xticks(ticks=np.arange(len(vms_to_compare))+0.5, labels=vms_to_compare, rotation=90, fontsize=18)
            plt.yticks(ticks=np.arange(len(vms_to_compare))+0.5, labels=vms_to_compare, rotation=0, fontsize=18)
            plt.title("", fontsize=18)
        st.pyplot(plt)
