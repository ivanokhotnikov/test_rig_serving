import io

import pandas as pd

from components.constants import RAW_DATA_BUCKET


def read_latest_unit(current_processed_df):
    latest = str(int(max(current_processed_df['UNIT'].unique())))
    for blob in RAW_DATA_BUCKET.list_blobs():
        if latest in blob.name and 'RAW' in blob.name:
            blob_bytes = blob.download_as_bytes()
            return pd.read_csv(
                io.BytesIO(blob_bytes),
                header=0,
                index_col=False,
            )
