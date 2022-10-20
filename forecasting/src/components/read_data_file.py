import io
import logging
import re

import pandas as pd

from components.constants import (FEATURES_NO_TIME, RAW_DATA_BUCKET,
                                  RAW_FORECAST_FEATURES)


def read_data_file(file):
    data_blob = RAW_DATA_BUCKET.get_blob(file.name)
    df = pd.read_csv(
        io.Bytes(data_blob.download_as_bytes()),
        usecols=RAW_FORECAST_FEATURES,
        index_col=False,
    )
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
    df['STEP'] = df['STEP'].astype(int)
    #TODO retesting case needs to be addressed
    df['TEST'] = int(1)
    return df
