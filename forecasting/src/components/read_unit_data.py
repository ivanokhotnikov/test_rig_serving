import io
import logging
import re

import pandas as pd
import streamlit as st
from components.constants import (FEATURES_NO_TIME, RAW_DATA_BUCKET,
                                  RAW_FORECAST_FEATURES)


@st.cache
def read_unit_data(file_name):
    data_blob = RAW_DATA_BUCKET.get_blob(file_name)
    df = pd.read_csv(
        io.BytesIO(data_blob.download_as_bytes()),
        index_col=False,
        header=0,
    )
    df = df.dropna(axis=0)
    try:
        unit = int(re.split(r'_|-|/', file_name)[0][-4:].lstrip('/HYDhyd0'))
        df['UNIT'] = unit
    except ValueError as err:
        logging.info(f'{err}\n. Cannot parse unit from {file_name}')
        df['UNIT'] = pd.NA
    return df
