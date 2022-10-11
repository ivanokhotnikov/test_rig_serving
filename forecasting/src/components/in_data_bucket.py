from components.constants import DATA_BUCKET


def in_data_bucket(file):
    raw_folder_content = {
        blob.name[4:]
        for blob in list(DATA_BUCKET.list_blobs(prefix='raw'))
    }
    return file in raw_folder_content
