from components.constants import RAW_DATA_BUCKET


def is_in_data_bucket(file):
    raw_folder_content = {
        blob.name
        for blob in list(RAW_DATA_BUCKET.list_blobs())
    }
    return file.name in raw_folder_content
