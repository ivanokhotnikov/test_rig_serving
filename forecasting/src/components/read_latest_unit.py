import io

import pandas as pd

from components.constants import DATA_BUCKET


def read_latest_unit(df):
    latest = str(int(max(df['UNIT'].unique())))
    for blob in DATA_BUCKET.list_blobs(prefix='raw'):
        if latest in blob.name and 'RAW' in blob.name:
            blob_bytes = blob.download_as_bytes()
            return pd.read_csv(
                io.BytesIO(blob_bytes),
                header=0,
                index_col=False,
            )
