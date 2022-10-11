import json

from components.constants import DATA_BUCKET


def import_final_features():
    data_blob = DATA_BUCKET.get_blob('features/final_features.json')
    return json.loads(data_blob.download_as_string())
