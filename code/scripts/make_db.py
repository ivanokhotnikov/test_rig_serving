import os
import sqlite3
import pandas as pd

from utils.readers import DataReader, Preprocessor
from utils.config import FEATURES

if __name__ == '__main__':
    os.chdir('..\\..')
    df = DataReader.read_all_raw_data(verbose=True, features_to_read=FEATURES)
    df = Preprocessor.remove_step_zero(df, inplace=False)
    df.sort_values(by=['DATE', 'TIME'], inplace=True, ignore_index=True)
    df['RUNNING TIME'] = pd.date_range(start=f'00:00:00 {df["DATE"].min()}',
                                       periods=len(df),
                                       freq='S')
    df.set_index('RUNNING TIME')
    df.to_sql('all_data',
              con=sqlite3.connect('data\\all_data.sqlite'),
              if_exists='replace',
              index=False)
