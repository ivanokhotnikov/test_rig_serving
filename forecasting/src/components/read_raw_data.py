import gc
import io
import logging

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
            if blob.name.endswith('.csv') and 'RAW' in blob.name:
                current_df = pd.read_csv(
                    io.BytesIO(data_bytes),
                    header=0,
                    index_col=False,
                )
            elif (blob.name.endswith('.xlsx')
                  or blob.name.endswith('.xls')) and 'RAW' in blob.name:
                current_df = pd.read_excel(
                    io.BytesIO(data_bytes),
                    header=0,
                    index_col=False,
                )
            else:
                logging.info(f'{blob.name} is not a valid raw data file.')
                continue
        except:
            logging.info(f'Can\'t read {blob.name}.')
            continue
        logging.info(f'{blob.name} has been read.')
        name_list = blob.name.split('-')
        try:
            unit = int(name_list[0][-3:].lstrip('0D'))
        except ValueError:
            unit = int(name_list[0].split('_')[0][-3:].lstrip('0D'))
        units.append(unit)
        current_df['UNIT'] = unit
        current_df['TEST'] = int(units.count(unit))
        final_df = pd.concat((final_df, current_df), ignore_index=True)
        del current_df
        gc.collect()
    try:
        final_df.sort_values(by=[' DATE', 'TIME'],
                             inplace=True,
                             ignore_index=True)
    except:
        logging.info('Can\'t sort dataframe')
    final_df[FEATURES_NO_TIME] = final_df[FEATURES_NO_TIME].apply(
        pd.to_numeric, errors='coerce', downcast='float')
    final_df.dropna(inplace=True)
    final_df = remove_step_zero(final_df)
    final_df = build_time_features(final_df)
    final_df = build_power_features(final_df)
    final_df.columns = final_df.columns.str.lstrip()
    final_df.columns = final_df.columns.str.replace(' ', '_')
    DATA_BUCKET.blob('processed/procssed_data.csv').upload_from_string(
        final_df.to_csv(index=False), content_type='text/csv')
    return final_df
