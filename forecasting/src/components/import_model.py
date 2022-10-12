import io
import gcsfs
import h5py
from joblib import load
from keras.models import load_model
from components.constants import MODELS_BUCKET


def import_model(file_name):
    if '.joblib' in file_name:
        blob = MODELS_BUCKET.get_blob(file_name)
        data_bytes = blob.download_as_bytes()
        return load(io.BytesIO(data_bytes))
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'gs://models_forecasting/{file_name}', 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        return load_model(model_gcs)