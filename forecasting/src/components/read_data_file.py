import io

import pandas as pd

from components.constants import DATA_BUCKET, FEATURES_NO_TIME, RAW_FORECAST_FEATURES


def read_data_file(file):
    data_blob = DATA_BUCKET.get_blob(f'raw/{file.name}')
    df = pd.read_csv(
            io.Bytes(data_blob.download_as_bytes()),
            usecols=RAW_FORECAST_FEATURES,
            index_col=False,
        )
    df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(pd.to_numeric,
                                                          errors='coerce',
                                                          downcast='float')
    df = df.dropna(axis=0)
    name_list = file.name.split('-')
    try:
        unit = int(name_list[0][-3:].lstrip('0D'))
    except ValueError:
        unit = int(name_list[0].split('_')[0][-3:].lstrip('0D'))
    df['UNIT'] = unit
    df['STEP'] = df['STEP'].astype(int)
    #TODO retesting case needs to be addressed
    df['TEST'] = int(1)
    return df
