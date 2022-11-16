from components.constants import RAW_DATA_BUCKET
import streamlit as st


def is_in_data_bucket(file):
    raw_folder_content = {
        blob.name
        for blob in list(RAW_DATA_BUCKET.list_blobs())
    }
    if not file.name in raw_folder_content:
        st.success('Not in the data storage', icon='✅')
    else:
        st.info('Already in the data storage', icon='ℹ️')
    return file.name in raw_folder_content
