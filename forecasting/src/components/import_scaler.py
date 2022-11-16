import io

from streamlit import cache
from components.constants import MODELS_BUCKET
from joblib import load


@cache(allow_output_mutation=True)
def import_scaler(file_name):
    blob = MODELS_BUCKET.get_blob(f'{file_name}.joblib')
    data_bytes = blob.download_as_bytes()
    joblib_model = load(io.BytesIO(data_bytes))
    return joblib_model