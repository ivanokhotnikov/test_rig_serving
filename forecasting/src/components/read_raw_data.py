import gc
import logging
import re

import pandas as pd
from components.build_power_features import build_power_features
from components.build_time_features import build_time_features
from components.constants import (FEATURES_NO_TIME, PROCESSED_DATA_BUCKET,
                                  RAW_DATA_BUCKET)
from components.is_name_valid import is_name_valid
from components.remove_step_zero import remove_step_zero

logging.basicConfig(level=logging.INFO)


def read_raw_data():
    """
    The read_raw_data function reads the raw data from the Google Cloud Storage bucket, cleans it and saves it as a parquet file in the processed data storage.
    
    Returns:
        A pandas dataframe
    """
    final_df = pd.DataFrame()
    units = []
    for blob in RAW_DATA_BUCKET.list_blobs():
        current_df = None
        try:
            if is_name_valid(blob) and blob.name.endswith('.csv'):
                current_df = pd.read_csv(f'gs://test_rig_raw_data/{blob.name}',
                                         header=0,
                                         index_col=False)
            elif (blob.name.endswith('.xlsx')
                  or blob.name.endswith('.xls')) and 'RAW' in blob.name[3:]:
                current_df = pd.read_excel(
                    f'gs://test_rig_raw_data/{blob.name}',
                    header=0,
                    index_col=False)
            else:
                logging.info(f'{blob.name} is not a valid raw data file')
                continue
        except:
            logging.info(f'Cannot read {blob.name}')
            continue
        logging.info(f'{blob.name} has been read')
        unit_str = re.split(r'_|-', blob.name.lstrip('/LN2'))[0][-4:]
        if unit_str == '':
            logging.info(f'Cannot parse unit from {blob.name}')
            continue
        unit = int(unit_str)
        units.append(unit)
        current_df['UNIT'] = unit
        current_df['TEST'] = int(units.count(unit))
        final_df = pd.concat((final_df, current_df), ignore_index=True)
        del current_df
        gc.collect()
    final_df.sort_values(by=['UNIT', 'TEST'], inplace=True, ignore_index=True)
    logging.info(f'Final dataframe sorted')
    final_df[FEATURES_NO_TIME] = final_df[FEATURES_NO_TIME].apply(
        pd.to_numeric, errors='coerce', downcast='float')
    final_df.dropna(subset=FEATURES_NO_TIME, axis=0, inplace=True)
    final_df.drop(
        columns=['DATE', ' DATE', 'DURATION', 'NOT USED', 'NOT_USED'],
        inplace=True,
        errors='ignore')
    logging.info(f'NAs, date and duration columns dropped')
    final_df = remove_step_zero(final_df)
    logging.info(f'Step zero removed')
    final_df = build_time_features(final_df)
    logging.info(f'Time features added')
    final_df = build_power_features(final_df)
    logging.info(f'Power features added')
    final_df.columns = final_df.columns.str.lstrip()
    final_df.columns = final_df.columns.str.replace(' ', '_')
    final_df.to_parquet('gs://test_rig_processed_data/processed_data.parquet',
                        index=False)
    logging.info(
        f'Processed dataframe uploaded to the {PROCESSED_DATA_BUCKET.name} data storage'
    )
    return final_df