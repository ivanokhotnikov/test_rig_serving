import io
import logging
import re

import pandas as pd
from components.constants import RAW_DATA_BUCKET
from streamlit import cache


@cache
def read_unit_data(file_name):
    """
    The read_unit_data function reads a CSV file from the raw data bucket, drops rows with missing values, and adds a column for the unit number.    
    
    Args:
        file_name: Get the unit number from the file name
    
    Returns:
        A pandas dataframe with the unit data
    """
    data_blob = RAW_DATA_BUCKET.get_blob(file_name)
    df = pd.read_csv(io.BytesIO(data_blob.download_as_bytes()),
                     index_col=False,
                     header=0)
    df = df.dropna(axis=0)
    try:
        unit = int(re.split(r'_|-|/', file_name)[0][-4:].lstrip('/HYDhyd0'))
        df['UNIT'] = unit
    except ValueError as err:
        logging.info(f'{err}\n. Cannot parse unit from {file_name}')
        df['UNIT'] = pd.NA
    return df
