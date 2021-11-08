import os
import numpy as np
import pandas as pd

from config import DATA_PATH, FEATURES, FEATURES_NO_TIME


class DataReader:
    def read_all_raw_data(self, verbose=True):
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
            name_list = file.split('-')
            try:
                unit = np.uint8(name_list[0][-2:])
            except ValueError:
                unit = np.uint8(name_list[0].split('_')[0][-2:])
            units.append(unit)
            current_df['ARMANI'] = 1 if name_list[0][3] == '2' else 0
            current_df['UNIT'] = unit
            current_df['TEST'] = np.uint8(units.count(unit))
            current_df['STEP'] = current_df['STEP'].astype(
                np.uint8,
                errors='ignore',
            )
            current_df = current_df.dropna(axis=0)
            final_df = pd.concat((final_df, current_df), ignore_index=True)
        return final_df

    def read_raw_unit_data(self, unit_id='HYD000091-R1_RAW'):
        try:
            unit_df = pd.read_csv(os.path.join(DATA_PATH, 'raw',
                                               unit_id + '.csv'),
                                  usecols=FEATURES,
                                  index_col=False,
                                  header=0)
        except:
            try:
                unit_df = pd.read_excel(os.path.join(DATA_PATH, 'raw',
                                                     unit_id + '.xlsx'),
                                        usecols=FEATURES,
                                        index_col=False,
                                        header=0)
            except:
                print(f'No {unit_id} file found')
                return None
        unit_df[FEATURES_NO_TIME] = unit_df[FEATURES_NO_TIME].apply(
            pd.to_numeric, errors='coerce')
        unit_df[FEATURES_NO_TIME] = unit_df[FEATURES_NO_TIME].astype(
            np.float32)
        unit_df = unit_df.dropna(axis=0)
        return unit_df

    def read_combined_data(self, verbose=True):
        try:
            if verbose:
                print('Reading "combined_data.csv"')
            df = pd.read_csv(os.path.join(os.getcwd(), DATA_PATH, 'processed',
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

    def read_summary_file(self, verbose=True):
        try:
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

    def load_data(self, read_all=True, raw=False, unit=None, verbose=True):
        if read_all:
            if raw:
                return self.read_all_raw_data(verbose=verbose)
            else:
                return self.read_combined_data(verbose=verbose)
        else:
            if raw:
                return self.read_raw_unit_data(unit_id=unit)
            else:
                return pd.DataFrame(self.read_summary_file())


class ModelReader:
    pass


class Preprocessor:
    def remove_outliers(self, df, zscore=3):
        from scipy import stats
        return df[(np.abs(stats.zscore(df[FEATURES_NO_TIME])) < zscore).all(
            axis=1)]

    def remove_step_zero(self, df):
        return df.drop(df[df['STEP'] == 0].index, axis=0)

    def get_warm_up_steps(self, df):
        return df[(df['STEP'] >= 1) & (df['STEP'] <= 11)]

    def get_break_in_steps(self, df):
        return df[(df['STEP'] >= 12) & (df['STEP'] <= 22)]

    def get_performance_check_steps(self, df):
        return df[(df['STEP'] >= 23) & (df['STEP'] <= 33)]


if __name__ == '__main__':
    os.chdir('../../..')
    os.getcwd()
    data_reader = DataReader()
    all_raw_data = data_reader.read_all_raw_data(verbose=True)
    all_combined_data = data_reader.read_combined_data(verbose=True)
    summary = data_reader.read_summary_file(verbose=True)
    combined_data = data_reader.load_data(raw=False)
    preprocessor = Preprocessor()
    wo_outliers = preprocessor.remove_outliers(combined_data, zscore=3)
    wo_step_zero = preprocessor.remove_step_zero(combined_data)
    warm_up = preprocessor.get_warm_up_steps(combined_data)
    break_in = preprocessor.get_break_in_steps(combined_data)
    performance_check = preprocessor.get_performance_check_steps(combined_data)
