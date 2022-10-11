from components.constants import DATA_BUCKET


def upload_new_raw_data_file(uploaded_file):
    new_data_blob = DATA_BUCKET.blob(f'raw/{uploaded_file.name}')
    new_data_blob.upload_from_file(
        uploaded_file,
        content_type='text/csv',
    )
