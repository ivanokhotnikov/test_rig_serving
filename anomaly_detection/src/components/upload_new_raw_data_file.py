from components.constants import RAW_DATA_BUCKET


def upload_new_raw_data_file(uploaded_file):
    """
    The upload_new_raw_data_file function uploads a new file to the raw data bucket. It takes as input an uploaded_file object, which is created by Streamlit's FileUploader. The function then creates a blob in the raw data bucket and uploads the contents of uploaded_file to it.
    
    Args:
        uploaded_file: Store the file that is uploaded by the user
    """
    new_data_blob = RAW_DATA_BUCKET.blob(uploaded_file.name)
    new_data_blob.upload_from_file(uploaded_file,
                                   content_type='text/csv',
                                   rewind=True)
