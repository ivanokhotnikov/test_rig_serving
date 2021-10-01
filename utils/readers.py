import os
import numpy as np
import pandas as pd
from . import config as c


def read_all_raw_data(verbose=False):
    final_df = pd.DataFrame()
    units = []
    for file in os.listdir('rig_data'):
        if verbose:
            print(f'Reading {file}')
        if file.endswith('.csv'):
            current_df = pd.read_csv(
                os.path.join(c.DATA_PATH, file),
                usecols=c.FEATURES,
                index_col=False,
                header=0,
            )
        elif file.endswith('.xlsx'):
            current_df = pd.read_excel(
                os.path.join(c.DATA_PATH, file),
                usecols=c.FEATURES,
                index_col=False,
                header=0,
            )
        current_df[c.FEATURES_NO_TIME] = current_df[c.FEATURES_NO_TIME].apply(
            pd.to_numeric,
            errors='coerce',
            downcast='float',
        )
        name_list = file.split('-')
        unit = np.uint8(name_list[0][-2:])
        units.append(unit)
        current_df['UNIT'] = unit
        current_df['TEST'] = np.uint8(
            units.count(unit))
        current_df['STEP'] = current_df['STEP'].astype(
            np.uint8,
            errors='ignore',
        )
        current_df = current_df.dropna(axis=0)
        final_df = pd.concat((final_df, current_df), ignore_index=True)
    return final_df


def read_raw_unit_data(unit_id='HYD000091-R1_RAW'):
    try:
        unit_df = pd.read_csv(os.path.join(c.DATA_PATH, unit_id + '.csv'),
                              usecols=c.FEATURES,
                              index_col=False,
                              header=0)
    except:
        try:
            unit_df = pd.read_excel(os.path.join(c.DATA_PATH,
                                                 unit_id + '.xlsx'),
                                    usecols=c.FEATURES,
                                    index_col=False,
                                    header=0)
        except:
            print(f'No {unit_id} file found')
            return None
    unit_df[c.FEATURES_NO_TIME] = unit_df[c.FEATURES_NO_TIME].apply(
        pd.to_numeric, errors='coerce')
    unit_df[c.FEATURES_NO_TIME] = unit_df[c.FEATURES_NO_TIME].astype(
        np.float32)
    unit_df = unit_df.dropna(axis=0)
    return unit_df


def read_combined_data():
    try:
        df = pd.read_csv('combined_df.csv',
                         usecols=c.FEATURES.append('UNIT'),
                         index_col=False)
        df[c.FEATURES_NO_TIME] = df[c.FEATURES_NO_TIME].apply(pd.to_numeric,
                                                              errors='coerce')
        df[c.FEATURES_NO_TIME] = df[c.FEATURES_NO_TIME].astype(np.float32)
        df[['STEP', 'UNIT', 'TEST']] = df[['STEP', 'UNIT', 'TEST']].astype(
            np.uint8,
            errors='ignore',
        )
        df = df.dropna(axis=0)
        return df
    except:
        print('No "combined_df.csv" found in the current directory')
        return None


def read_summary_file():
    try:
        xl = pd.ExcelFile('report template-V4.xlsx')
        units = {}
        for sheet in xl.sheet_names:
            if 'HYD' in sheet:
                units[f'{sheet}'] = pd.read_excel(xl, sheet_name=sheet)
        return units
    except:
        print('No "report template-V4.xlsx" found')
        return None


def load_data(read_all=True, raw=False, unit=None):
    if read_all:
        if raw:
            return read_all_raw_data()
        else:
            return read_combined_data()
    else:
        if raw:
            return read_raw_unit_data(unit_id=unit)
        else:
            return pd.DataFrame(read_summary_file())