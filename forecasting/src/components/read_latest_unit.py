import pandas as pd
from components.constants import RAW_DATA_BUCKET
from streamlit import cache


@cache
def read_latest_unit(current_processed_df):
    """
    The read_latest_unit function reads the latest unit from the raw data bucket. It does this by finding the most recent unit in that bucket and then reading it into a pandas dataframe.
    
    Args:
        current_processed_df: Pass the dataframe that is currently being processed
    
    Returns:
        The latest unit that has been processed
    """
    latest = str(int(max(current_processed_df['UNIT'].unique())))
    for blob in RAW_DATA_BUCKET.list_blobs():
        if latest in blob.name and 'RAW' in blob.name:
            return pd.read_csv(f'gs://test_rig_raw_data/{blob.name}',
                               header=0,
                               index_col=False)