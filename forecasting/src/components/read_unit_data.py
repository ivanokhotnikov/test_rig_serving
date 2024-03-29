import logging
import re

import pandas as pd


def read_unit_data(file_name):
    """
    The read_unit_data function reads a CSV file from the raw data bucket, drops rows with missing values, and adds a column for the unit number.    
    
    Args:
        file_name: Get the unit number from the file name
    
    Returns:
        A pandas dataframe with the unit data
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(f'gs://test_rig_raw_data/{file_name}',
                            index_col=False,
                            header=0)
        elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(f'gs://test_rig_raw_data/{file_name}',
                            index_col=False,
                            header=0)
        else:
            logging.info(f'{file_name} is not a valid raw data file')
            return None
    except:
        logging.info(f'Cannot read {file_name}')
        return None
    df = df.dropna(axis=0)
    try:
        unit = int(re.split(r'_|-', file_name.lstrip('/LN2'))[0][-4:])
        df['UNIT'] = unit
    except ValueError as err:
        logging.info(f'{err}\n. Cannot parse unit from {file_name}')
        df['UNIT'] = pd.NA
    return df
