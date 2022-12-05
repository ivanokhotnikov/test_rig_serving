import pandas as pd


def build_time_features(df):
    """
    The build_time_features function takes a dataframe as input and returns a new dataframe with two additional features: RUNNING_SECONDS, which is the number of seconds since the start of the race, and RUNNING_HOURS, which is RUNNING_SECONDS divided by 3600.
    
    Args:
        df: Pass the dataframe to the function
    
    Returns:
        A dataframe with two new features: running_seconds and running_hours
    """
    df['RUNNING_SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(int)
    df['RUNNING_HOURS'] = (df['RUNNING_SECONDS'] / 3600).astype(float)
    return df
