import json

from components.constants import MODELS_BUCKET


def import_metrics(feature):
    data_blob = MODELS_BUCKET.get_blob(f'{feature}.json')
    return json.loads(data_blob.download_as_string())
