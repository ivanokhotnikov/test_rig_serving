def upload_processed_data(df):
    """
    The upload_processed_data function uploads the processed data to a GCS bucket. The function takes in a Pandas DataFrame as an argument and returns nothing.
    
    Args:
        df: Pass the dataframe to be uploaded
    """
    df.to_csv('gs://test_rig_processed_data/processed_data.csv', index=False)
