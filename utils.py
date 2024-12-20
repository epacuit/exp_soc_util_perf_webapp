import streamlit as st
import pandas as pd
from google.cloud import storage
import io
import zipfile

def load_csv_from_gcs(bucket_name: str, file_path: str, file_size: int) -> pd.DataFrame:
    """
    Load a zipped CSV file from a public Google Cloud Storage bucket directly into a Pandas DataFrame.
    Display a progress bar to indicate the download progress.
    """
    
    try:
        # Step 1: Create an anonymous storage client and get the blob reference
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        # Step 2: Initialize progress bar and download status
        progress_bar = st.progress(0)
        status_container = st.empty()
        file_content = bytearray()

        # Step 3: Download the file in chunks
        with blob.open("rb") as f:
            chunk_size = 1024 * 1024  # 1 MB chunks
            total_downloaded = 0

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                file_content.extend(chunk)
                total_downloaded += len(chunk)
                progress = min(100, int((total_downloaded / file_size) * 100))
                progress_bar.progress(progress)
                status_container.text(f'Downloading {progress}% complete ({total_downloaded / 1024 / 1024:.2f} MB)')

        progress_bar.empty()
        status_container.empty()

        # Step 4: Extract the CSV file from the downloaded zip archive
        with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zf:
            # Assuming there is only one CSV file in the zip, extract the first file
            csv_filename = zf.namelist()[0] 
            with zf.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)  # Read the CSV file directly into a DataFrame

        return df

    except Exception as e:
        st.error(f"An error occurred while downloading or extracting the file: {e}")
        return pd.DataFrame()  

@st.cache_resource(show_spinner="Loading main simulation data...")
def load_main_dataframe():
    if "main_data_frame" not in st.session_state:
        file_size_mb = 247.9

        file_size = int(file_size_mb * (1024 * 1024))
        df = load_csv_from_gcs("soc-util-perf-data", "exp_soc_util_perf_data.csv.zip", file_size)

        df = df[df['vm'] != "Random Dictator"]
        df = df[df['vm'] != "Proportional Borda"]

        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)

        df['vm'] = df['vm'].replace('PluralityWRunoff PUT', 'Plurality with Runoff')
        df['vm'] = df['vm'].replace('Blacks', "Black's")
        df['vm'] = df['vm'].replace('Bottom-Two-Runoff Instant Runoff', 'Bottom-Two-Runoff IRV')
        df['vm'] = df['vm'].replace('Tideman Alternative Top Cycle', 'Tideman Alternative Smith')

        st.session_state['main_data_frame'] = df
    else: 
        df = st.session_state['main_data_frame']

    all_voting_methods = sorted(df['vm'].unique())
    polarized_df = df[df['num_dims_polarized'] != 0]
    unpolarized_df = df[df['num_dims_polarized'] == 0]

    return polarized_df, unpolarized_df, all_voting_methods

@st.cache_resource(show_spinner="Loading uncertainty simulation data...")
def load_uncertainty_dataframe():
    if "uncertainty_data_frame" not in st.session_state:

        file_size_mb = 182.1

        # Convert MB to bytes
        file_size = int(file_size_mb * (1024 * 1024))
        df = load_csv_from_gcs("soc-util-perf-data", "exp_soc_util_perf_data_uncertainty.csv.zip", file_size)

        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)

        df['vm'] = df['vm'].replace('PluralityWRunoff PUT', 'Plurality with Runoff')
        df['vm'] = df['vm'].replace('Blacks', "Black's")
        df['vm'] = df['vm'].replace('Bottom-Two-Runoff Instant Runoff', 'Bottom-Two-Runoff IRV')
        df['vm'] = df['vm'].replace('Tideman Alternative Top Cycle', 'Tideman Alternative Smith')

        st.session_state['uncertainty_data_frame'] = df
    else: 
        df = st.session_state['uncertainty_data_frame']

    all_voting_methods = sorted(df['vm'].unique())
    polarized_df = df[df['num_dims_polarized'] != 0]
    unpolarized_df = df[df['num_dims_polarized'] == 0]

    return polarized_df, unpolarized_df, all_voting_methods

@st.cache_resource(show_spinner="Loading Condorcet efficiency and absolute utility data...")
def load_condorcet_efficiency_data(filename):

    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            # Get list of files in the zip, filtering for CSV files
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV files found in the ZIP archive.")
            
            # Extract and load the first CSV file
            with zip_ref.open(csv_files[0]) as file:
                df = pd.read_csv(file)
    else:
        df = pd.read_csv(filename)

    return df
