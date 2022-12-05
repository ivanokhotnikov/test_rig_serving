from components.constants import PROCESSED_DATA_BUCKET


def upload_processed_data(df):
    """
    The upload_processed_data function uploads the processed data to a GCS bucket. The function takes in a Pandas DataFrame as an argument and returns nothing.
    
    Args:
        df: Pass the dataframe to be uploaded
    """
    updated_processed_data_blob = PROCESSED_DATA_BUCKET.blob(
        'processed_data.csv')
    updated_processed_data_blob.upload_from_string(df.to_csv(index=False),
                                                   content_type='text/csv')
