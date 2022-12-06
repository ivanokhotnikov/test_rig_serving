import pandas as pd
from streamlit import cache


@cache
def read_processed_data():
    """
    The read_processed_data function reads the processed data from a CSV file in a GCS bucket. The function returns the data as a Pandas DataFrame.
      
    Returns:
        A pandas dataframe
    """
    return pd.read_csv('gs://test_rig_processed_data/processed_data.csv',
                       header=0,
                       index_col=False,
                       low_memory=False)
