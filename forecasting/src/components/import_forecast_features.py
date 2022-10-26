import json

from components.constants import FEATURES_BUCKET


def import_forecast_features():
    data_blob = FEATURES_BUCKET.get_blob('forecast_features.json')
    return json.loads(data_blob.download_as_bytes())
