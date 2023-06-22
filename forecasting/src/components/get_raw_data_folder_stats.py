def get_raw_data_folder_stats():
    """
    The get_raw_data_folder_stats function returns the number of files in the raw data folder and 
    the number of valid files.    
        
    Returns:
        A tuple of the number of files and the number of valid files in the raw data folder
    """
    from components.is_name_valid import is_name_valid
    from components.constants import RAW_DATA_BUCKET
    num_files = len([True for _ in RAW_DATA_BUCKET.list_blobs()])
    num_valid_files = len(
        [True for blob in RAW_DATA_BUCKET.list_blobs() if is_name_valid(blob)])
    return num_files, num_valid_files
