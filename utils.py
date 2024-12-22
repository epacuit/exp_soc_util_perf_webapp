import streamlit as st
import pandas as pd
from google.cloud import storage
import io
import zipfile
import requests
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
                df = pd.read_csv(csv_file, chunksize=100_000)  # Read the CSV file directly into a DataFrame

        return df

    except Exception as e:
        st.error(f"An error occurred while downloading or extracting the file: {e}")
        return pd.DataFrame()  

@st.cache_resource(show_spinner="Loading main simulation data...")
def load_main_dataframe():
    if "main_data_frame" not in st.session_state:
        file_size_mb = 247.9
        file_size = int(file_size_mb * (1024 * 1024))
        
        # Set the chunk size for processing the CSV in smaller pieces
        chunksize = 10_000
        
        # Create an empty list to store processed chunks
        filtered_chunks = []

        # Step 1: Download the ZIP file and extract the specific file
        url = 'https://storage.googleapis.com/soc-util-perf-data/exp_soc_util_perf_data.csv.zip'
        response = requests.get(url)
        
        #st.write(f"Downloaded {len(response.content) / 1024 / 1024:.2f} MB")  # Debugging statement
        # Open the ZIP file from the in-memory response
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # List all files in the ZIP
            file_names = z.namelist()
            print(f"Files in ZIP: {file_names}")  # Debugging statement
            
            # Extract only the CSV file we want
            target_file = 'exp_soc_util_perf_data.csv'
            
            if target_file not in file_names:
                raise ValueError(f"File {target_file} not found in ZIP. Available files: {file_names}")
            
            with z.open(target_file) as csvfile:
                # Step 2: Read and process CSV in chunks
                for chunk in pd.read_csv(csvfile, chunksize=chunksize):
                    # Filter out "Random Dictator" and "Proportional Borda"
                    chunk = chunk[chunk['vm'] != "Random Dictator"]
                    chunk = chunk[chunk['vm'] != "Proportional Borda"]

                    # Drop Unnamed: 0 if it exists in this chunk
                    if "Unnamed: 0" in chunk.columns:
                        chunk.drop("Unnamed: 0", axis=1, inplace=True)

                    # Rename 'vm' values
                    chunk['vm'] = chunk['vm'].replace({
                        'PluralityWRunoff PUT': 'Plurality with Runoff',
                        'Blacks': "Black's",
                        'Bottom-Two-Runoff Instant Runoff': 'Bottom-Two-Runoff IRV',
                        'Tideman Alternative Top Cycle': 'Tideman Alternative Smith'
                    })

                    # Store the processed chunk
                    filtered_chunks.append(chunk)
        
        # Concatenate all filtered chunks into one DataFrame
        df = pd.concat(filtered_chunks, ignore_index=True)

        # Store the DataFrame in session state to avoid reloading
        st.session_state['main_data_frame'] = df
    else: 
        df = st.session_state['main_data_frame']

    # Extract unique voting methods
    all_voting_methods = sorted(df['vm'].unique())
    
    # Create two subsets for polarized and unpolarized data
    polarized_df = df[df['num_dims_polarized'] != 0]
    unpolarized_df = df[df['num_dims_polarized'] == 0]

    return polarized_df, unpolarized_df, all_voting_methods

@st.cache_resource(show_spinner="Loading uncertainty simulation data...")
def load_uncertainty_dataframe():
    if "uncertainty_data_frame" not in st.session_state:

        file_size_mb = 182.1
        # Create an empty list to store processed chunks
        filtered_chunks = []
        chunksize = 10_000

        url = 'https://storage.googleapis.com/soc-util-perf-data/exp_soc_util_perf_data_uncertainty.csv.zip'
        response = requests.get(url)
        
        #st.write(f"Downloaded {len(response.content) / 1024 / 1024:.2f} MB")  # Debugging statement

            
        # Open the ZIP file from the in-memory response
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # List all files in the ZIP
            file_names = z.namelist()
            print(f"Files in ZIP: {file_names}")  # Debugging statement

            csv_files = [f for f in z.namelist() if not f.startswith('__MACOSX/') and not f.startswith('._')]
            if len(csv_files) != 1:
                raise ValueError(f"Expected one file in ZIP, but found {csv_files}")
            
            with z.open(csv_files[0]) as csvfile:
                # Step 2: Read and process CSV in chunks
                for chunk in pd.read_csv(csvfile, chunksize=chunksize):
                    # Filter out "Random Dictator" and "Proportional Borda"
                    chunk = chunk[chunk['vm'] != "Random Dictator"]
                    chunk = chunk[chunk['vm'] != "Proportional Borda"]

                    # Drop Unnamed: 0 if it exists in this chunk
                    if "Unnamed: 0" in chunk.columns:
                        chunk.drop("Unnamed: 0", axis=1, inplace=True)

                    # Rename 'vm' values
                    chunk['vm'] = chunk['vm'].replace({
                        'PluralityWRunoff PUT': 'Plurality with Runoff',
                        'Blacks': "Black's",
                        'Bottom-Two-Runoff Instant Runoff': 'Bottom-Two-Runoff IRV',
                        'Tideman Alternative Top Cycle': 'Tideman Alternative Smith'
                    })

                    # Store the processed chunk
                    filtered_chunks.append(chunk)
        
        # Concatenate all filtered chunks into one DataFrame
        df = pd.concat(filtered_chunks, ignore_index=True)


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
            csv_files = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX/') and not f.startswith('._')]
            if len(csv_files) != 1:
                raise ValueError(f"Expected one file in ZIP, but found {csv_files}")
            
            # Extract and load the first CSV file
            with zip_ref.open(csv_files[0]) as file:
                df = pd.read_csv(file)
    else:
        df = pd.read_csv(filename)

    return df
