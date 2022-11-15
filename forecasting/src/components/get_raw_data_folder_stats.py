import streamlit as st


@st.cache
def get_raw_data_folder_stats():
    from components import is_name_valid
    from components.constants import RAW_DATA_BUCKET
    num_files = len([True for _ in RAW_DATA_BUCKET.list_blobs()])
    num_valid_files = len(
        [True for blob in RAW_DATA_BUCKET.list_blobs() if is_name_valid(blob)])
    return num_files, num_valid_files
