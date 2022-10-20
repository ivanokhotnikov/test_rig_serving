from components.constants import PROCESSED_DATA_BUCKET


def upload_processed_data(df):
    updated_processed_data_blob = PROCESSED_DATA_BUCKET.blob(
        'processed_data.csv')
    updated_processed_data_blob.upload_from_string(
        df.to_csv(index=False),
        content_type='text/csv',
    )
