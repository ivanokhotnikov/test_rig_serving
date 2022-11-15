import io

import pandas as pd
import streamlit as st
from components.constants import PROCESSED_DATA_BUCKET


@st.cache
def read_processed_data():
    data_blob = PROCESSED_DATA_BUCKET.get_blob('processed_data.csv')
    return pd.read_csv(
        io.BytesIO(data_blob.download_as_bytes()),
        header=0,
        index_col=False,
        low_memory=False,
    )