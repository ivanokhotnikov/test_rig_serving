import io

from components.constants import MODELS_BUCKET
from joblib import load

def import_scaler(file_name):
    """
    The import_scaler function downloads a pre-trained scaler from the GCS bucket and returns it as an object. The function takes one argument, which is the name of the file containing the scaler.
    
    Args:
        file_name: Specify the name of the file that is being imported
    
    Returns:
        The model that was saved in the blob
    """
    blob = MODELS_BUCKET.get_blob(f'{file_name}.joblib')
    data_bytes = blob.download_as_bytes()
    joblib_model = load(io.BytesIO(data_bytes))
    return joblib_model