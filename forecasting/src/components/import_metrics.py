import json

from components.constants import MODELS_BUCKET
from streamlit import cache


@cache(allow_output_mutation=True)
def import_metrics(feature):
    """
    The import_metrics function downloads the JSON file from the models bucket and returns a dictionary of metrics. The function takes one argument, feature, which is used to determine which file to download.
    
    Args:
        feature: Specify which feature we are interested in
    
    Returns:
        A dictionary of the model's metrics
    """
    data_blob = MODELS_BUCKET.get_blob(f'{feature}.json')
    return json.loads(data_blob.download_as_bytes())