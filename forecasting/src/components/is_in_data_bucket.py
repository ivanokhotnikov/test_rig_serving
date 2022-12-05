from components.constants import RAW_DATA_BUCKET
import streamlit as st


def is_in_data_bucket(file):
    """
    The is_in_data_bucket function checks if the file is already in the data storage. It returns True if it's there, False otherwise.
    
    Args:
        file: Check if the file is already in the data storage
    
    Returns:
        A boolean value

    """
    raw_folder_content = {
        blob.name
        for blob in list(RAW_DATA_BUCKET.list_blobs())
    }
    if not file.name in raw_folder_content:
        st.success('Not in the data storage', icon='✅')
    else:
        st.info('Already in the data storage', icon='ℹ️')
    return file.name in raw_folder_content