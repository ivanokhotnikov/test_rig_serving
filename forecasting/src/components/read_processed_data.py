import io

import pandas as pd
from components.constants import PROCESSED_DATA_BUCKET


def read_processed_data():
    """
    The read_processed_data function reads the processed data from a CSV file in a GCS bucket. The function returns the data as a Pandas DataFrame.
      
    Returns:
        A pandas dataframe
    """
    data_blob = PROCESSED_DATA_BUCKET.get_blob('processed_data.csv')
    return pd.read_csv(io.BytesIO(data_blob.download_as_bytes()),
                       header=0,
                       index_col=False,
                       low_memory=False)
