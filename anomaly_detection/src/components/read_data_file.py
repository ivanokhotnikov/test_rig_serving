import io
import logging
import re

import pandas as pd
from components.constants import (FEATURES_NO_TIME, RAW_DATA_BUCKET,
                                  RAW_FORECAST_FEATURES)


def read_data_file(file):
    """
    The read_data_file function reads a CSV file from the raw data bucket, applies the preprocessing: convert to numeric, drop na values, parse unit number from file.name, and returns a Pandas DataFrame. The function accepts one argument: `file`, which is an object representing a single blob in our raw data bucket.
    
    Args:
        file: Pass the name of the file to be read
    
    Returns:
        A pandas dataframe object
    """
    data_blob = RAW_DATA_BUCKET.get_blob(file.name)
    df = pd.read_csv(io.BytesIO(data_blob.download_as_string()),
                     usecols=RAW_FORECAST_FEATURES,
                     index_col=False)
    df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(pd.to_numeric,
                                                      errors='coerce',
                                                      downcast='float')
    df = df.dropna(axis=0)
    try:
        unit = int(re.split(r'_|-|/', file.name)[0][-4:].lstrip('/HYDhyd0'))
        df['UNIT'] = unit
    except ValueError as err:
        logging.info(f'{err}\n. Cannot parse unit from {file.name}')
        df['UNIT'] = pd.NA
    return df
