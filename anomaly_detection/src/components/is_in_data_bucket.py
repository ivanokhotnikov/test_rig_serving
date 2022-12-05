from components.constants import RAW_DATA_BUCKET


def is_in_data_bucket(file):
    """
    The is_in_data_bucket function checks if a file is in the raw data bucket.
    
    Args:
        file: Check if a file exists in the raw data bucket
    
    Returns:
        True if the file is in the raw data bucket
    """
    raw_folder_content = {
        blob.name
        for blob in list(RAW_DATA_BUCKET.list_blobs())
    }
    return file.name in raw_folder_content
