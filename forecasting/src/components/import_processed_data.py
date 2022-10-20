import io

import pandas as pd

from components.constants import PROCESSED_DATA_BUCKET


def import_processed_data():
    data_blob = PROCESSED_DATA_BUCKET.get_blob('processed_data.csv')
    data_bytes = data_blob.download_as_bytes()
    return pd.read_csv(
        io.BytesIO(data_bytes),
        header=0,
        index_col=False,
        low_memory=False,
    )
