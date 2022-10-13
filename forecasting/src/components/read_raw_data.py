import gc
import io
import logging
import re

import pandas as pd
import streamlit as st


def read_raw_data():
    from components import (build_power_features, build_time_features,
                            remove_step_zero)
    from components.constants import (DATA_BUCKET, DATA_BUCKET_NAME,
                                      FEATURES_NO_TIME)

    final_df = pd.DataFrame()
    units = []
    st.write(f'Reading raw data files from {DATA_BUCKET_NAME}')
    loading_bar = st.progress(0)
    for idx, blob in enumerate(list(DATA_BUCKET.list_blobs(prefix='raw')), 1):
        loading_bar.progress(idx /
                             len(list(DATA_BUCKET.list_blobs(prefix='raw'))))
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
            unit = int(re.split(r'_|-|/', blob.name)[1][4:].lstrip('/HYD0'))
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
    logging.info(f'NAs, date and duration columns droped')
    final_df = remove_step_zero(final_df)
    logging.info(f'Step zero removed')
    final_df = build_time_features(final_df)
    logging.info(f'Time features added')
    final_df = build_power_features(final_df)
    logging.info(f'Power features added')
    final_df.columns = final_df.columns.str.lstrip()
    final_df.columns = final_df.columns.str.replace(' ', '_')
    DATA_BUCKET.blob('processed/processed_data.csv').upload_from_string(
        final_df.to_csv(index=False), content_type='text/csv')
    logging.info(f'Processed dataframe uploaded to data storage')
    return final_df
