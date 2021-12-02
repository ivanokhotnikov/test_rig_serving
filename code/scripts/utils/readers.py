import os
import numpy as np
import pandas as pd
import streamlit as st

from .config import DATA_PATH, MODELS_PATH, FEATURES, FEATURES_NO_TIME
from joblib import load


class DataReader:
    @staticmethod
    def read_all_raw_data(verbose=True):
        final_df = pd.DataFrame()
        units = []
        for file in os.listdir(os.path.join(DATA_PATH, 'raw')):
            if verbose:
                print(f'Reading {file}')
            try:
                if file.endswith('.csv'):
                    current_df = pd.read_csv(
                        os.path.join(DATA_PATH, 'raw', file),
                        usecols=FEATURES,
                        on_bad_lines='skip',
                        index_col=False,
                        header=0,
                    )
                elif file.endswith('.xlsx'):
                    current_df = pd.read_excel(
                        os.path.join(DATA_PATH, 'raw', file),
                        usecols=FEATURES,
                        index_col=False,
                        header=0,
                    )
            except ValueError:
                if verbose:
                    print(f'{file} got a faulty header')
                continue
            current_df[FEATURES_NO_TIME] = current_df[FEATURES_NO_TIME].apply(
                pd.to_numeric,
                errors='coerce',
                downcast='float',
            )
            current_df = current_df.dropna(axis=0)
            name_list = file.split('-')
            try:
                unit = np.uint8(name_list[0][-3:].lstrip('0D'))
            except ValueError:
                unit = np.uint8(name_list[0].split('_')[0][-3:].lstrip('0D'))
            units.append(unit)
            current_df['ARMANI'] = 1 if name_list[0][3] == '2' else 0
            current_df['ARMANI'] = current_df['ARMANI'].astype(np.uint8)
            current_df['UNIT'] = unit
            current_df['TEST'] = np.uint8(units.count(unit))
            current_df['STEP'] = current_df['STEP'].astype(np.uint8)
            final_df = pd.concat((final_df, current_df), ignore_index=True)
        if verbose: print('Reading done!')
        return final_df

    @staticmethod
    def read_raw_unit_data(unit='HYD000091-R1_RAW'):
        try:
            unit_df = pd.read_csv(os.path.join(DATA_PATH, 'raw',
                                               unit + '.csv'),
                                  usecols=FEATURES,
                                  index_col=False,
                                  header=0)
        except:
            try:
                unit_df = pd.read_excel(os.path.join(DATA_PATH, 'raw',
                                                     unit + '.xlsx'),
                                        usecols=FEATURES,
                                        index_col=False,
                                        header=0)
            except:
                print(f'No {unit} file found')
                return None
        unit_df[FEATURES_NO_TIME] = unit_df[FEATURES_NO_TIME].apply(
            pd.to_numeric, errors='coerce')
        unit_df[FEATURES_NO_TIME] = unit_df[FEATURES_NO_TIME].astype(
            np.float32)
        unit_df = unit_df.dropna(axis=0)
        return unit_df

    @staticmethod
    def read_combined_data(verbose=True):
        try:
            if verbose:
                print('Reading "combined_data.csv"')
            df = pd.read_csv(os.path.join(DATA_PATH, 'processed',
                                          'combined_data.csv'),
                             usecols=FEATURES.append('UNIT'),
                             index_col=False)
            if verbose:
                print('Processing data')
            df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(pd.to_numeric,
                                                              errors='coerce')
            df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].astype(np.float32)
            df[['STEP', 'UNIT', 'TEST']] = df[['STEP', 'UNIT', 'TEST']].astype(
                np.uint8,
                errors='ignore',
            )
            df = df.dropna(axis=0)
            if verbose:
                print('Done')
            return df
        except:
            print('No "combined_data.csv" found in the "data" directory')
            return None

    @staticmethod
    def read_summary_data(verbose=True):
        try:
            if verbose:
                print(f'Reading the summsry file.')
            xl = pd.ExcelFile(
                os.path.join(DATA_PATH, 'processed',
                             'report template-V4.xlsx'))
            units = {}
            for sheet in xl.sheet_names:
                if 'HYD' in sheet:
                    if verbose:
                        print(f'Reading {sheet}')
                    units[f'{sheet}'] = pd.read_excel(xl, sheet_name=sheet)
            if verbose:
                print(f'Done')
            return units
        except:
            print('No "report template-V4.xlsx" found')
            return None

    @classmethod
    def load_data(cls, read_all=True, raw=False, unit=None, verbose=True):
        if read_all:
            if raw:
                return cls.read_all_raw_data(verbose=verbose)
            else:
                return cls.read_combined_data(verbose=verbose)
        else:
            if raw:
                return cls.read_raw_unit_data(unit=unit)
            else:
                return pd.DataFrame(cls.read_summary_data())

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def read_newcoming_data(csv_file):
        df = pd.read_csv(csv_file,
                         usecols=FEATURES,
                         on_bad_lines='skip',
                         index_col=False,
                         header=0)
        df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(
            pd.to_numeric,
            errors='coerce',
            downcast='float',
        )
        df = df.dropna(axis=0)
        name_list = csv_file.name.split('-')
        try:
            unit = np.uint8(name_list[0][-3:].lstrip('0D'))
        except ValueError:
            unit = np.uint8(name_list[0].split('_')[0][-3:].lstrip('0D'))
        df['UNIT'] = unit
        df['STEP'] = df['STEP'].astype(np.uint8)
        df['TEST'] = np.uint8(1)
        return df


class Preprocessor:
    @staticmethod
    def remove_outliers(df, zscore=3):
        from scipy import stats
        return df[(np.abs(stats.zscore(df[FEATURES_NO_TIME])) < zscore).all(
            axis=1)]

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def remove_step_zero(df):
        return df.drop(df[df['STEP'] == 0].index, axis=0)

    @staticmethod
    def get_warm_up_steps(df):
        return df[(df['STEP'] >= 1) & (df['STEP'] <= 11)]

    @staticmethod
    def get_break_in_steps(df):
        return df[(df['STEP'] >= 12) & (df['STEP'] <= 22)]

    @staticmethod
    def get_performance_check_steps(df):
        return df[(df['STEP'] >= 23) & (df['STEP'] <= 33)]


class ModelReader:
    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def read_model(model, extension='joblib'):
        return load(os.path.join(MODELS_PATH, f'{model}' + '.' + extension))


if __name__ == '__main__':
    os.chdir('../../..')
    os.getcwd()
    all_raw_data = DataReader.read_all_raw_data(verbose=True)
    all_raw_data['STEP'].unique()
    all_combined_data = DataReader.read_combined_data(verbose=True)
    summary = DataReader.read_summary_data(verbose=True)
    combined_data = DataReader.load_data(raw=False)
    wo_outliers = Preprocessor.remove_outliers(combined_data, zscore=3)
    wo_step_zero = Preprocessor.remove_step_zero(combined_data)
    warm_up = Preprocessor.get_warm_up_steps(combined_data)
    break_in = Preprocessor.get_break_in_steps(combined_data)
    performance_check = Preprocessor.get_performance_check_steps(combined_data)
    model = ModelReader.read_model('isolation_forest')
