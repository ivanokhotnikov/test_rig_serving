import gc
import io
import logging
import re

import pandas as pd
import streamlit as st


def read_raw_data():
    from components import (build_power_features, build_time_features,
                            remove_step_zero)
    from components.constants import (RAW_DATA_BUCKET, PROCESSED_DATA_BUCKET,
                                      INTERIM_DATA_BUCKET, FEATURES_NO_TIME)

    final_df = pd.DataFrame()
    units = []
    st.write(f'Reading raw data files from {RAW_DATA_BUCKET.name}')
    loading_bar = st.progress(0)
    for idx, blob in enumerate(list(RAW_DATA_BUCKET.list_blobs()), 1):
        loading_bar.progress(idx / len(list(RAW_DATA_BUCKET.list_blobs())))
        data_bytes = blob.download_as_bytes()
        current_df = None
        try:
            if blob.name.endswith('.csv') and 'RAW' in blob.name[3:]:
                current_df = pd.read_csv(
                    io.BytesIO(data_bytes),
                    header=0,
                    index_col=False,
                )
            elif (blob.name.endswith('.xlsx')
                  or blob.name.endswith('.xls')) and 'RAW' in blob.name[3:]:
                current_df = pd.read_excel(
                    io.BytesIO(data_bytes),
                    header=0,
                    index_col=False,
                )
            else:
                logging.info(f'{blob.name} is not a valid raw data file')
                continue
        except:
            logging.info(f'Cannot read {blob.name}')
            continue
        logging.info(f'{blob.name} has been read')
        try:
            unit = int(
                re.split(r'_|-|/', blob.name)[0][-4:].lstrip('/HYDhyd0'))
        except ValueError as err:
            logging.info(f'{err}\n. Cannot parse unit from {blob.name}')
            continue
        units.append(unit)
        current_df['UNIT'] = unit
        current_df['TEST'] = int(units.count(unit))
        final_df = pd.concat((final_df, current_df), ignore_index=True)
        del current_df
        gc.collect()
    try:
        final_df.sort_values(
            by=['UNIT', 'TEST'],
            inplace=True,
            ignore_index=True,
        )
        logging.info(f'Final dataframe sorted')
    except:
        logging.info('Cannot sort dataframe')
    INTERIM_DATA_BUCKET.blob('interim_data.csv').upload_from_string(
        final_df.to_csv(index=False), content_type='text/csv')
    logging.info(
        f'Interim dataframe uploaded to the {INTERIM_DATA_BUCKET.name} data storage'
    )
    final_df[FEATURES_NO_TIME] = final_df[FEATURES_NO_TIME].apply(
        pd.to_numeric,
        errors='coerce',
        downcast='float',
    )
    final_df.dropna(
        subset=FEATURES_NO_TIME,
        axis=0,
        inplace=True,
    )
    final_df.drop(columns='DATE', inplace=True, errors='ignore')
    final_df.drop(columns=' DATE', inplace=True, errors='ignore')
    final_df.drop(columns='DURATION', inplace=True, errors='ignore')
    logging.info(f'NAs, date and duration columns dropped')
    final_df = remove_step_zero(final_df)
    logging.info(f'Step zero removed')
    final_df = build_time_features(final_df)
    logging.info(f'Time features added')
    final_df = build_power_features(final_df)
    logging.info(f'Power features added')
    final_df.columns = final_df.columns.str.lstrip()
    final_df.columns = final_df.columns.str.replace(' ', '_')
    PROCESSED_DATA_BUCKET.blob('processed_data.csv').upload_from_string(
        final_df.to_csv(index=False), content_type='text/csv')
    logging.info(
        f'Processed dataframe uploaded to the {PROCESSED_DATA_BUCKET.name} data storage'
    )
    return final_df
