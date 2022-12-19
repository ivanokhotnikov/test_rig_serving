import json

from components.constants import FEATURES_BUCKET

def import_forecast_features():
    """
    The import_forecast_features function downloads the forecast_features.json file from the GCS feature store bucket and returns it as a dictionary.
    
    Args:
    
    Returns:
        A dictionary of the forecast features
    """
    data_blob = FEATURES_BUCKET.get_blob('forecast_features.json')
    return json.loads(data_blob.download_as_bytes())
