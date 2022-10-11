from components.constants import DATA_BUCKET


def upload_data(df, data_type):
    updated_processed_data_blob = DATA_BUCKET.blob(
        f'{data_type}/{data_type}_data.csv')
    updated_processed_data_blob.upload_from_string(
        df.to_csv(index=False),
        content_type='text/csv',
    )
