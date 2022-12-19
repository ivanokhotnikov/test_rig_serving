import pandas as pd


def read_processed_data(features: list):
    """
    The read_processed_data function reads in the processed data from a Parquet file. The function takes as input a list of features to be read, and returns a Pandas DataFrame containing those features.
    
    Args:
        features: list: Specify which columns to read from the parquet file
    
    Returns:
        A pandas dataframe with the specified features
    """
    return pd.read_parquet(
        'gs://test_rig_processed_data/processed_data.parquet',
        columns=features)
