import streamlit as st

from pref_voting.voting_methods import * 
from pref_voting.generate_spatial_profiles import *   
from pref_voting.utility_functions import *
from pref_voting.probabilistic_methods import *
from pref_voting.profiles import *
from pref_voting.utility_methods import *

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D, art3d

import math

st.set_page_config(
    page_title="Visualization of Spatial Profiles",
    page_icon="📊",
)

def expected_utility(prob, util_func):
    return sum([prob[c] * util_func(c) for c in prob.keys()])

def to_linear_prof(uprof): 
    return Profile([sorted(uprof.domain, key=lambda x: u(x), reverse=True) for u in uprof.utilities])

def find_winning_probs(vms, prob_vms, prof): 
    prob_ws =  {vm.name: vm.prob(prof) for vm in vms}
    for vm in prob_vms:
        prob_ws[vm.name] = vm(prof)
    return prob_ws

def soc_util_surplus(
    avg_util_of_util_ws, 
    avg_util_of_cand, 
    avg_util_of_vm_ws):

    return (avg_util_of_vm_ws - avg_util_of_cand) / (avg_util_of_util_ws - avg_util_of_cand)


st.title("Visualization of Spatial Profiles")

all_num_cands =   [3, 4, 5, 6, 7, 8, 9, 10] 
all_num_voters = [11, 101, 1001]
all_num_dims =    [1, 2, 3]
all_normalizations = ["none", "range"]
all_is_polarized = [False, True]
all_num_centrist_cands = ["none", "half", "all"]
all_num_dims_polarized = ["one", "half", "all"]
all_prob_centrist_voters = [0.0, 0.5, 1.0]
all_polarization_distances = [1]
all_subpopulation_stds = [1, 0.5]
all_dispersion =  [1, 0.5]
all_correlation = [0, 0.5]
all_voter_utilities = {
    "Linear": linear_utility, 
    "Quadratic": quadratic_utility, 
    "Shepsle": shepsle_utility,
    "Matthews": matthews_utility,
    "Mixed Proximity-RM": mixed_rm_utility,
    "RM": rm_utility    
    }

vms = [
    condorcet,
    condorcet_plurality,
    copeland_global_minimax,
    copeland,
    copeland_local_borda,
    copeland_global_borda,
    plurality,
    anti_plurality,
    borda,
    instant_runoff,
    plurality_with_runoff_put,
    benham, 
    bottom_two_runoff_instant_runoff,
    coombs,
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
    bucklin,
    woodall, 
    river_zt,
    smith_minimax, 
    tideman_alternative_smith
]

condorcet_vms = [
    condorcet,
    copeland,
    copeland_local_borda,
    copeland_global_borda,
    #plurality,
    #anti_plurality,
    #borda,
    #instant_runoff,
    #plurality_with_runoff_put,
    benham, 
    bottom_two_runoff_instant_runoff,
    #coombs,
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
    #bucklin,
    woodall, 
    river_zt,
    smith_minimax, 
    tideman_alternative_smith
]


prob_vms = [random_dictator, pr_borda]

num_cands_to_num_centrists = {nc: list(set([{"none": 0, "one": 1, "half": nc // 2, "all": nc}[cent] for cent in all_num_centrist_cands])) for nc in all_num_cands}
print(num_cands_to_num_centrists)

dim_to_num_dims_polarized = {nd: sorted([_n for _n in list(set([{"one": 1, "half": nd // 2, "all": nd}[pol] for pol in all_num_dims_polarized])) if _n > 0]) for nd in all_num_dims}

with st.sidebar:
    st.header("Choose Parameters")
    st.subheader("Choose Parameters for Spatial Profile")
    
    num_cands = st.select_slider("Number of Candidates", all_num_cands, value=5)

    num_voters  = st.select_slider("Number of Voters", all_num_voters)
        
    num_dims = st.radio(
        "Number of Dimensions", all_num_dims, index=1 )
        
    rel_dispersion = st.radio(
        "Relative Dispersion", all_dispersion)
        
    correlation = st.radio(
        "Correlation", all_correlation )
        
    is_polarized = st.checkbox("Polarization")
    if is_polarized: 
        polarization_distance = 1

        num_dims_polarized = st.radio(
            "Number of Polarized Dimensions", dim_to_num_dims_polarized[num_dims])

        subpopulation_std = st.radio(
            "Subpopulation Standard Deviation", all_subpopulation_stds)

        num_centrist_cands = st.radio(
            "Number of Centrist Candidates",sorted(num_cands_to_num_centrists[num_cands]))

        prob_centrist_voters = st.radio(
            "Probability of Centrist Voter", all_prob_centrist_voters if num_centrist_cands != num_cands else all_prob_centrist_voters[0:2] )

    else: 
        num_dims_polarized = 0
        polarization_distance = 0
        subpopulation_std = 1
        num_centrist_cands = 0
        prob_centrist_voters = 0.0

    st.subheader("Choose Parameters for Utility Profile")

    normalization = st.radio(
        "Normalization", all_normalizations)

    voter_utility = st.radio(
        "Voter Utility", 
        list(all_voter_utilities.keys()))


    include_probabilistic_vms = st.checkbox(
        "Include Probabilistic Voting Methods", False)

def generate_spatial_profile_randomly_polarized_voters(
        num_polarized_cands, 
        num_centrist_cands,
        num_voters,
        prob_centrist_voter, 
        num_dims, 
        cand_cov,
        voter_cov,
        num_dims_polarized,
        polarization_distance): 
    
    cand_cov = cand_cov if cand_cov is not None else np.eye(num_dims)
    voter_cov = voter_cov if voter_cov is not None else np.eye(num_dims)

    num_left_cands = num_polarized_cands // 2
    sprof = generate_spatial_profile_polarized_cands_randomly_polarized_voters(
        [
            (np.array([polarization_distance] * num_dims_polarized +  [0] * (num_dims - num_dims_polarized)), cand_cov, num_left_cands),

            (np.array([-1 * polarization_distance] * num_dims_polarized + [0] * (num_dims - num_dims_polarized)), cand_cov, num_polarized_cands - num_left_cands),

            (np.array([0] * num_dims), cand_cov, num_centrist_cands)],
        num_voters,
        [
            (np.array([polarization_distance] * num_dims_polarized + [0] * (num_dims - num_dims_polarized)), voter_cov, (1 - prob_centrist_voter) / 2),

            (np.array([-1 * polarization_distance] * num_dims_polarized + [0] * (num_dims - num_dims_polarized)), voter_cov, (1 - prob_centrist_voter) / 2), 
            
            (np.array([0] * num_dims), voter_cov, prob_centrist_voter)],
             
        num_profiles = 1)

    return sprof

@st.cache_data
def get_spatial_profile(
        num_cands, 
        num_voters, 
        num_dims,
        rel_dispersion,
        correlation,
        is_polarized,
        num_dims_polarized,
        polarization_distance,
        subpopulation_std,
        num_centrist_cands,
        prob_centrist_voters):
    

    if not is_polarized:
        if rel_dispersion == 1: 
            cand_cov = generate_covariance(num_dims, 1, correlation)
            voter_cov = cand_cov
        elif rel_dispersion == 0.5:
            cand_cov = generate_covariance(num_dims, 0.5, correlation)
            voter_cov = generate_covariance(num_dims, 1, correlation)
                
    elif is_polarized:
        if rel_dispersion == 1: 
            voter_cov = generate_covariance(num_dims, subpopulation_std, correlation)
            cand_cov = voter_cov
        elif rel_dispersion == 0.5:
            voter_cov = generate_covariance(num_dims, subpopulation_std, correlation)
            cand_cov = generate_covariance(num_dims, 0.5 * subpopulation_std, correlation)

    return generate_spatial_profile_randomly_polarized_voters(                    num_cands - num_centrist_cands, 
        num_centrist_cands,
        num_voters,
        prob_centrist_voters, 
        num_dims, 
        cand_cov,
        voter_cov,
        num_dims_polarized,
        polarization_distance)


sprof = get_spatial_profile(
    num_cands, 
    num_voters, 
    num_dims,
    rel_dispersion,
    correlation,
    is_polarized,
    num_dims_polarized,
    polarization_distance,
    subpopulation_std,
    num_centrist_cands,
    prob_centrist_voters)
    

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Spatial Profile", 
    "Voter Distribution", 
    "Utility Functions",
    "Candidate Utilities", 
    "Social Utility Performance"])

with tab1: 
    if st.button("Generate Spatial Profile", key="button_tab1"):
        st.cache_data.clear()


    if num_dims == 1: 

        # Extract the positions for a single dimension (e.g., dim1)
        voter_df = pd.DataFrame({
            "position": [sprof.voter_position(v)[0] for v in sprof.voters], 
            "Type": ["Voter"] * len(sprof.voters)
        })

        cand_df = pd.DataFrame({
            "position": [sprof.candidate_position(c)[0] for c in sprof.candidates], 
            "Type": ["Candidate"] * len(sprof.candidates)
        })

        # Combine DataFrames, ensuring candidates are plotted last
        position_df = pd.concat([voter_df, cand_df])

        # Calculate x-axis range to fit data tightly
        min_pos = position_df['position'].min()
        max_pos = position_df['position'].max()
        padding = 0.05 * (max_pos - min_pos)

        # Create the scatter plot
        fig = px.scatter(
            position_df,
            x='position',
            y=[0] * len(position_df),  # Keep everything on the same line
            color='Type',
            symbol='Type',
            color_discrete_map={
                "Voter": "steelblue",
                "Candidate": "orange"
            },
            opacity=0.7,
            size_max=10  # Increase marker size
        )

        # Update layout for better visibility
        fig.update_traces(marker=dict(size=12))  # Larger markers
        fig.update_layout(
            yaxis=dict(visible=False, showgrid=False, zeroline=False),   
            xaxis=dict(
                title='Dimension 1',
                showgrid=False,
                zeroline=False,
                range=[min_pos - padding, max_pos + padding]  # Tighter range
            ),
            height=100,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    elif num_dims == 2:
        max_pos = math.ceil(max([math.fabs(sprof.voter_position(v)[0]) for v in sprof.voters] + [math.fabs(sprof.voter_position(v)[1]) for v in sprof.voters]))

        voter_df = pd.DataFrame({
            "dim1-position": [sprof.voter_position(v)[0] for v in sprof.voters], 
            "dim2-position": [sprof.voter_position(v)[1] for v in sprof.voters],
            "Type": ["Voter"] * len(sprof.voters)
        })
                        
        cand_df = pd.DataFrame({
            "dim1-position": [sprof.candidate_position(c)[0] for c in sprof.candidates], 
            "dim2-position": [sprof.candidate_position(c)[1] for c in sprof.candidates],
            "Type": ["Candidate"] * len(sprof.candidates)
        })

        position_df = pd.concat([voter_df, cand_df])
                        
        voter_plot = alt.Chart(position_df).mark_point(
            filled=True,
            size=100
        ).encode(
            # x=alt.X("dim1-position", title="Dimension 1", scale=alt.Scale(domain=[-max_pos, max_pos]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),

            # y=alt.Y("dim2-position", title="Dimension 2", scale=alt.Scale(domain=[-max_pos, max_pos]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),

            x=alt.X("dim1-position", title="Dimension 1", scale=alt.Scale(domain=[-4, 4]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),

            y=alt.Y("dim2-position", title="Dimension 2", scale=alt.Scale(domain=[-4, 4]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),

            color=alt.value('steelblue'),        

            tooltip=[alt.Tooltip('dim1-position', format=".3f", title="Dimension 1 Position"), alt.Tooltip('dim2-position', format=".3f", title="Dimension 2 Position")],
            ).transform_filter(
                alt.datum.Type == 'Voter'
            )
        candidate_plot = alt.Chart(position_df).mark_square(
            filled=True,
            opacity=1.0,
            size=100
        ).encode(
            #x=alt.X("dim1-position", title="Dimension 1", scale=alt.Scale(domain=[-max_pos, max_pos]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),

            #y=alt.Y("dim2-position", title="Dimension 2", scale=alt.Scale(domain=[-max_pos, max_pos]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),
            x=alt.X("dim1-position", title="Dimension 1", scale=alt.Scale(domain=[-4, 4]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),

            y=alt.Y("dim2-position", title="Dimension 2", scale=alt.Scale(domain=[-4, 4]), axis=alt.Axis(tickMinStep=1, tickCount=max_pos * 2 + 1, titleFontSize=16)),

            color=alt.value('orange'),        
            tooltip=[alt.Tooltip('dim1-position', format=".3f", title="Dimension 1 Position"), alt.Tooltip('dim2-position', format=".3f", title="Dimension 2 Position")],
            ).transform_filter(
                alt.datum.Type == 'Candidate'
            )

        legend_data = pd.DataFrame({
            'Type': ['Voter', 'Candidate'],
            'color': ['steelblue', 'orange'],
            'shape': ['circle', 'square']
        })

        legend = alt.Chart(legend_data).mark_point(filled=True, size=100).encode(
            y=alt.Y('Type:N', axis=alt.Axis(orient='right', title='', labelFontSize=16, domain=False, ticks=False, grid=False), scale=alt.Scale(domain=['Voter', 'Candidate'])),
            color=alt.Color('color:N', scale=None),
            shape=alt.Shape('shape:N', scale=None),
            tooltip=alt.value(None)
        ).properties(
            width=50,
            height=50
        )

        # Combine the plots
        scatter_plot = alt.layer(voter_plot, candidate_plot).properties(
            title=" ",
            height=600,
            width=600
        )

        # Combine the scatter plot with the legend
        final_plot = alt.hconcat(scatter_plot, legend).resolve_legend(
            color='independent',
            shape='independent'
        )


        st.subheader("Candidate and Voter Positions")

        st.altair_chart(final_plot, use_container_width=True)

    elif num_dims == 3:

        # Calculate the maximum absolute value for scaling axes
        max_pos = math.ceil(max(
            [abs(sprof.voter_position(v)[i]) for v in sprof.voters for i in range(3)] + 
            [abs(sprof.candidate_position(c)[i]) for c in sprof.candidates for i in range(3)]
        ))

        # Add a small buffer to max_pos to avoid cutting off the axes
        buffer = 0.1 * max_pos
        adjusted_max_pos = max_pos + buffer

        # Create DataFrame for voters
        voter_df = pd.DataFrame({
            "dim1-position": [sprof.voter_position(v)[0] for v in sprof.voters], 
            "dim2-position": [sprof.voter_position(v)[1] for v in sprof.voters],
            "dim3-position": [sprof.voter_position(v)[2] for v in sprof.voters],
            "Type": ["Voter"] * len(sprof.voters)
        })

        # Create DataFrame for candidates
        cand_df = pd.DataFrame({
            "dim1-position": [sprof.candidate_position(c)[0] for c in sprof.candidates], 
            "dim2-position": [sprof.candidate_position(c)[1] for c in sprof.candidates],
            "dim3-position": [sprof.candidate_position(c)[2] for c in sprof.candidates],
            "Type": ["Candidate"] * len(sprof.candidates)
        })

        # Combine DataFrames
        position_df = pd.concat([voter_df, cand_df])

        # Create the 3D scatter plot
        fig = px.scatter_3d(
            position_df, 
            x='dim1-position', 
            y='dim2-position', 
            z='dim3-position',
            color='Type',
            symbol='Type',
            color_discrete_map={
                "Voter": "steelblue",
                "Candidate": "orange"
            },
            opacity=0.7
        )

        # Increase width and height to accommodate the high scale factor
        fig.update_layout(
            width=500,  # Large enough to handle scale=5
            height=600,
            autosize=False,
            scene=dict(
                xaxis=dict(title='Dimension 1', autorange=True, zeroline=False),
                yaxis=dict(title='Dimension 2', autorange=True, zeroline=False),
                zaxis=dict(title='Dimension 3', autorange=True, zeroline=False),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)  # Adjust aspect ratio to give more space on the left
            ),
            margin=dict(l=100, r=0, t=0, b=0)  # Increased left margin to prevent cut-off
            )

        fig3 = plt.figure()
        ax = fig3.add_subplot(projection='3d')

        # Plot voters
        ax.scatter(
            [sprof.voter_position(v)[0] for v in sprof.voters], 
            [sprof.voter_position(v)[1] for v in sprof.voters], 
            [sprof.voter_position(v)[2] for v in sprof.voters], 
            color='steelblue', alpha=0.5, label='Voters'
        )

        # Plot candidates
        ax.scatter(
            [sprof.candidate_position(c)[0] for c in sprof.candidates], 
            [sprof.candidate_position(c)[1] for c in sprof.candidates], 
            [sprof.candidate_position(c)[2] for c in sprof.candidates], 
            marker='s', color='orange', label='Candidates'
        )
        # Function to draw a sphere at a given position
        def plot_sphere(ax, center, radius=0.1, color='steelblue', alpha=0.5):
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_surface(x, y, z, color=color, shade=True, alpha=alpha)

        # Function to draw a cube at a given position
        def plot_cube(ax, center, size=0.2, color='orange', alpha=0.8):
            r = [-size/2, size/2]
            # Define the vertices of a cube
            vertices = np.array([[x, y, z] for x in r for y in r for z in r])
            # Define the 6 faces of the cube
            faces = [[vertices[j] for j in [0, 1, 3, 2]],
                    [vertices[j] for j in [4, 5, 7, 6]],
                    [vertices[j] for j in [0, 2, 6, 4]],
                    [vertices[j] for j in [1, 3, 7, 5]],
                    [vertices[j] for j in [0, 1, 5, 4]],
                    [vertices[j] for j in [2, 3, 7, 6]]]
            
            # Shift the cube to the center and draw each face without black edges
            for face in faces:
                face_vertices = np.array(face) + center
                poly3d = art3d.Poly3DCollection([face_vertices], color=color, alpha=alpha)
                ax.add_collection3d(poly3d)

        # Increase the figure size to ensure enough space
        fig3 = plt.figure(figsize=(12, 10))
        ax = fig3.add_subplot(projection='3d')

        # Plot spheres for voters with higher transparency
        for v in sprof.voters:
            plot_sphere(ax, sprof.voter_position(v), alpha=0.3)

        # Plot cubes for candidates without black edges
        for c in sprof.candidates:
            plot_cube(ax, sprof.candidate_position(c), alpha=0.8)

        # Set axis labels with increased padding
        ax.set_xlabel('Dimension 1', labelpad=10)
        ax.set_ylabel('Dimension 2', labelpad=10)
        ax.set_zlabel('Dimension 3', labelpad=10)

        # Set the background of each axis pane to white
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Customize gridlines to be gray
        ax.xaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)
        ax.yaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)
        ax.zaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)

        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

        # # Adjust the subplot parameters to ensure labels are fully visible
        # plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)


        st.plotly_chart(fig, use_container_width=True)



with tab2: 

    if num_dims == 1 or num_dims == 3:
        st.warning("The voter distribution can only be visualized in 2 dimensions. Please select 2 dimensions for the voter distribution.")
    else: 
        voter_means = [np.array([polarization_distance] * num_dims_polarized + [0] * (num_dims - num_dims_polarized)), np.array([-1 * polarization_distance] * num_dims_polarized + [0] * (num_dims - num_dims_polarized)), np.array([0, 0])]

        if not is_polarized:
            if rel_dispersion == 1: 
                cand_cov = generate_covariance(num_dims, 1, correlation)
                voter_cov = cand_cov
            elif rel_dispersion == 0.5:
                cand_cov = generate_covariance(num_dims, 0.5, correlation)
                voter_cov = generate_covariance(num_dims, 1, correlation)
                    
        elif is_polarized:
            if rel_dispersion == 1: 
                voter_cov = generate_covariance(num_dims, subpopulation_std, correlation)
                cand_cov = voter_cov
            elif rel_dispersion == 0.5:
                voter_cov = generate_covariance(num_dims, subpopulation_std, correlation)
                cand_cov = generate_covariance(num_dims, 0.5 * subpopulation_std, correlation)

        x, y = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))

        # Calculate values of the two bivariate normal distributions
        rv1 = multivariate_normal(voter_means[0], voter_cov)
        rv2 = multivariate_normal(voter_means[1], voter_cov)
        rv3 = multivariate_normal(voter_means[2], voter_cov)

        pdf1 = rv1.pdf(pos)
        pdf2 = rv2.pdf(pos)
        pdf3 = rv3.pdf(pos)

        weight1 = (1 - prob_centrist_voters) / 2
        weight2 = (1 - prob_centrist_voters) / 2
        weight3 = prob_centrist_voters

        mixture_pdf = weight1 * pdf1 + weight2 * pdf2 + weight3 * pdf3

        fig = go.Figure(data=[go.Surface(z=mixture_pdf, x=x, y=y, colorscale='Viridis', showscale=False)])

        fig.update_layout(
            title='',
                scene=dict(
                        xaxis=dict(title='X'),
                        yaxis=dict(title='Y'),
                        zaxis=dict(title='')
                    ),
                width=600,  # Set the desired width
                height=600   # Set the desired height
            )
        

        st.subheader("Probability Distribution Generating the Voter Positions")

        st.plotly_chart(fig)

        fig.update_layout(
        scene=dict(
            xaxis=dict(range=[3, -3], title='X'),
            yaxis=dict(range=[-3, 3], title='Y'),
            zaxis=dict(title='Probability Density'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Set camera position
            )
        )
        )

with tab3: # utility functions

    x_values = np.linspace(-4, 4, 100)
    y_values = np.linspace(-4, 4, 100)
    candidate_positions = list(product(x_values, y_values))

    all_voter_utilities = {
        "Linear": linear_utility, 
        "Quadratic": quadratic_utility, 
        "Shepsle": shepsle_utility,
        "Matthews": matthews_utility,
        "Mixed Proximity-RM": mixed_rm_utility,
        "RM": rm_utility,
        #"City Block": city_block_utility,    
        }
    # select voter utility function
    st.write("Select a voter utility function")
    voter_util = st.selectbox("", list(all_voter_utilities.keys()), key="voter_util", label_visibility="collapsed")

    st.write("Select a voter position")

    left, right = st.columns(2, vertical_alignment="bottom")
    voter_pos_y = left.slider("Voter Position on Dimension 1",min_value=-4.0, max_value=4.0, value=0.0, step=0.1, key="voter_pos_y")

    voter_pos_x = right.slider("Voter Position on Dimension 2",min_value=-4.0, max_value=4.0, value=0.0, step=0.1, key="voter_pos_x")

    if voter_pos_x == 0.0 and voter_pos_y == 0.0 and voter_util == "Matthews":
        st.warning("The Matthews utility function is not defined when the voter position is at the origin. Please move the voter position to visualize the utility.")
    else:
        voter_pos = (float(voter_pos_x), float(voter_pos_y))
        v_util_fn = all_voter_utilities[voter_util]

        utils = [v_util_fn(np.array(voter_pos), np.array([x_pos, y_pos])) for x_pos in x_values for y_pos in y_values]

        utils_matrix = np.array(utils).reshape(len(x_values), len(y_values))

        fig = go.Figure(data=go.Heatmap(
            z=utils_matrix,
            x=x_values,
            y=y_values,
            colorscale='Viridis'
        ))

        fig.add_trace(go.Scatter(
            x=[voter_pos[1]],
            y=[voter_pos[0]],
            mode='markers',
            marker=dict(size=10, color='black'),
            name='Voter Position'
        ))

        fig.update_layout(
            title=f'{voter_util} Utility Function',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            autosize=False,
            width=600,
            height=600,
            margin=dict(l=50, r=50, b=50, t=50)
        )

        st.plotly_chart(fig, key='voter')


with tab4:
    if st.button("Generate Spatial Profile", key="button_tab4"):
        st.cache_data.clear()

    sprof = get_spatial_profile(
            num_cands, 
            num_voters, 
            num_dims,
            rel_dispersion,
            correlation,
            is_polarized,
            num_dims_polarized,
            polarization_distance,
            subpopulation_std,
            num_centrist_cands,
            prob_centrist_voters)

    _uprof = sprof.to_utility_profile(utility_function=all_voter_utilities[voter_utility])
    utilities = _uprof.utilities

    if normalization == 'range':
        uprof = _uprof.normalize_by_range()  
    elif normalization == 'score':
        uprof = _uprof.normalize_by_standard_score() 
    else: 
        uprof = _uprof

    dfs = []
    for c in uprof.domain:
        candidate_data = pd.DataFrame({
            'Utility': [u(c) for u in uprof.utilities],
            'Candidate': [f'Candidate {c}' for _ in uprof.utilities]
        })
        dfs.append(candidate_data)

    # Combine data for all candidates
    df = pd.concat(dfs)
    fig, ax = plt.subplots()
    sns.violinplot(x='Utility', y='Candidate', data=df, inner='quartile', ax=ax)
    #ax.set_title('')
    ax.set_xlabel('Utility Value')
    ax.set_ylabel('')
    
    # despine ax
    sns.despine(ax=ax, left=True, bottom=True, top=True, right=True)

    st.subheader("Distribution of Utilities for Each Candidate")
    st.pyplot(fig)

    _uprof = sprof.to_utility_profile(utility_function=all_voter_utilities[voter_utility])
    if normalization == 'range':
        uprof = _uprof.normalize_by_range()  
    elif normalization == 'score':
        uprof = _uprof.normalize_by_standard_score() 
    else: 
        uprof = _uprof

    u_ws = sum_utilitarian(uprof)

    st.markdown("### All Winning Sets")

    if len(u_ws) == 1:
        st.markdown(f"The utilitarian winner is  **Candidate {u_ws[0]}**")
    elif len(u_ws) == 2:
        st.markdown(f"The utilitarian winners are  **Candidate {u_ws[0]} and Candidate{u_ws[1]}**")
    else: 
        cand_str = ''
        for c in u_ws[0:-1]: 
            cand_str += f"Candidate {c}, "
        cand_str += f"and Candidate {u_ws[-1]}"
        st.markdown(f"The utilitarian winners are **{cand_str}**")
    #st.write(df)
    prof = to_linear_prof(uprof)
    cw = prof.condorcet_winner()
    if cw is not None:
        st.markdown(f"The Condorcet winner is **Candidate {cw}**")

    winning_sets = {}

    for vm in vms: 
        ws = vm(prof)
        if tuple(ws) not in winning_sets.keys():
            winning_sets[tuple(ws)] = [vm.name]
        else:
            winning_sets[tuple(ws)].append(vm.name)

    for ws in winning_sets.keys():

        if len(ws) == 1:
            ws_str = f"Candidate {ws[0]}"
        if len(ws) == 2:
            ws_str = f"Candidate {ws[0]} and Candidate {ws[1]}"
        if len(ws) > 2:
            ws_str = ''
            for c in ws[0:-1]: 
                ws_str += f"Candidate {c}, "
            ws_str += f"and Candidate {ws[-1]}"

        if len(winning_sets[ws]) == 1:
            st.write(f"{len(winning_sets[ws])} voting method selects the winning set: **{ws_str}**")
        else:
            st.write(f"{len(winning_sets[ws])} voting methods select the winning set: **{ws_str}**")
        st.markdown(f"* {', '.join(winning_sets[ws])}")

    vm_name_to_vm = {vm.name: vm for vm in vms + prob_vms}
    vm_name = st.selectbox("Select a Voting Method", sorted([vm.name for vm in vms + prob_vms]))

    vm = vm_name_to_vm[vm_name]
    if vm in prob_vms:
        pr_ws = vm(to_linear_prof(uprof))
        for c,pr in pr_ws.items():
            st.markdown(f"Probability of Candidate {c} Winning: **{pr}**")
    else: 
        ws = vm(to_linear_prof(uprof))
        print(ws)
        if len(ws) == 1:
            st.markdown(f"{vm.name} Winner: **Candidate {ws[0]}**")
        else: 
            cand_str = ''
            for c in ws[0:-1]: 
                cand_str += f"Candidate {c}, "    
            cand_str += f"Candidate {ws[-1]}"
            st.markdown(f"{vm.name} Winners: **{cand_str}**")

with tab5:
    if st.button("Generate Spatial Profile", key="button_tab5"):
        st.cache_data.clear()
    
    all_voting_methods = sorted([vm.name for vm in vms + prob_vms])  

    selected_vms = st.multiselect(
        'Select Voting Methods', 
        options=all_voting_methods,
        default=sorted([
            "Plurality", 
            "Condorcet Plurality",
            "Instant Runoff",
            "Condorcet IRV",
            "Borda", 
            #"Approval", 
            "Copeland-Global-Borda", 
            "Copeland-Global-Minimax", 
            #"Plurality with Runoff", 
            "Minimax", 
            "Stable Voting", 
            "Beat Path", 
            "Ranked Pairs ZT", 
            "Split Cycle",
            "Blacks",
            ]))

    _uprofs = {
        voter_utility: sprof.to_utility_profile(utility_function=all_voter_utilities[voter_utility]) for voter_utility in all_voter_utilities.keys()
    }
    if normalization == 'range':
        uprofs = {voter_util: _up.normalize_by_range() for voter_util,_up in _uprofs.items()} 
    else: 
        uprofs = _uprofs

    profs = {
        voter_util: to_linear_prof(uprof) 
        for voter_util, uprof in uprofs.items()}
    
    candidates = list(range(num_cands))

    util_ws_s = {
        voter_util: sum_utilitarian(uprof) 
        for voter_util, uprof in uprofs.items()}
    
    avg_utils = {
        voter_util: uprof.avg_utility_function()
        for voter_util, uprof in uprofs.items()}
    
    winning_prob_dicts  =  {
        voter_util: find_winning_probs(vms, prob_vms, prof)
        for voter_util, prof in profs.items()}
        
    avg_util_of_util_ws =  {
        voter_util:  np.average([avg_utils[voter_util](w) for w in util_ws]) for voter_util, util_ws in util_ws_s.items()}

    avg_util_of_cand = {
        voter_util: np.average([avg_util(c) for c in candidates]) 
        for voter_util, avg_util in avg_utils.items()}

    soc_util_perfs = {voter_util: 
     {vm.name: soc_util_surplus(avg_util_of_util_ws[voter_util], avg_util_of_cand[voter_util], expected_utility(winning_prob_dicts[voter_util][vm.name], avg_utils[voter_util])) 
      for vm in vms + prob_vms} 
      for voter_util in  all_voter_utilities.keys()}


    categories = list(all_voter_utilities.keys())
    num_methods = len(selected_vms)
    bar_width = 0.1

    # Create the figure
    fig = go.Figure()

    # Add horizontal bars for each category
    for i, category in enumerate(categories):
        scores = [soc_util_perfs[category][method] for method in selected_vms]
        fig.add_trace(
            go.Bar(
                x=scores,
                y=[method for method in selected_vms],
                name=category,
                orientation='h',
                width=bar_width,
                offset=i * bar_width - (len(categories) * bar_width) / 2,  # Center bars
            )
        )

    fig.update_layout(
        #title="Social Utility Performance",
        xaxis=dict(
            title="", 
            showgrid=True, 
            gridcolor='lightgray',
            tickfont=dict(size=16), 
            zeroline=False),
        yaxis=dict(
            title="", 
            showgrid=False, 
            tickmode='array', 
            tickvals=list(range(num_methods)), 
            ticktext=selected_vms,
            tickfont=dict(size=16),  # Increase font size for y-axis labels
            ticks='outside',
            automargin=True
        ),
        barmode='overlay',  # Ensures bars are overlaid with some offset
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=16),
            title='',
        ),
        height=800,  # Adjust as needed for your content
    )
    st.subheader("Social Utility Performance")
    st.plotly_chart(fig, use_container_width=True)

