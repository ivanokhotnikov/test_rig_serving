import json

from components.constants import MODELS_BUCKET
from streamlit import cache


@cache(allow_output_mutation=True)
def import_metrics(feature):
    data_blob = MODELS_BUCKET.get_blob(f'{feature}.json')
    return json.loads(data_blob.download_as_bytes())
