from components.constants import RAW_DATA_BUCKET


def upload_new_raw_data_file(uploaded_file):
    new_data_blob = RAW_DATA_BUCKET.blob(uploaded_file.name)
    new_data_blob.upload_from_file(
        uploaded_file,
        content_type='text/csv',
        rewind=True,
    )
