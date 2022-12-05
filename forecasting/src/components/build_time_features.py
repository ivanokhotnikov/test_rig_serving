import pandas as pd
from streamlit import cache


@cache
def build_time_features(df):
    """
    The build_time_features function adds a few new columns to the dataframe. The RUNNING_SECONDS column is simply the number of seconds that have elapsed since the beginning of the run. The RUNNING_HOURS column is then calculated by dividing the total number of seconds by 3600, which converts it into hours.
    
    Args:
        df: Pass the data frame that we are working on
    
    Returns:
        A dataframe with two new columns: running_seconds, running_hours
    """
    df['RUNNING_SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(int)
    df['RUNNING_HOURS'] = (df['RUNNING_SECONDS'] / 3600).astype(float)
    return df
