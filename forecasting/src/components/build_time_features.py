import pandas as pd

def build_time_features(df):
    """
    The build_time_features function takes a dataframe as input and returns a new dataframe with two additional features:
        1. RUNNING_SECONDS - the number of seconds since the beginning of the dataset
        2. RUNNING_HOURS - the number of hours since the beginning of the dataset
    
    Args:
        df: Pass the dataframe to the function
    
    Returns:
        A dataframe with two new columns, running_seconds and running_hours
    """
    df['TIME'] = df['TIME'].astype(str)
    df['RUNNING_SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(int)
    df['RUNNING_HOURS'] = (df['RUNNING_SECONDS'] / 3600).astype(float)
    return df
