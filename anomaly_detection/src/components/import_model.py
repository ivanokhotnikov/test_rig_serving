import io
import gcsfs
import h5py
from joblib import load
from keras.models import load_model
from components.constants import MODELS_BUCKET


def import_model(model):
    if 'Autoencoder' in model:
        if 'scaler' in model:
            blob = MODELS_BUCKET.get_blob(model + '.joblib')
            return load(io.BytesIO(blob.download_as_bytes()))
        if 'threshold' in model:
            blob = MODELS_BUCKET.get_blob(model + '.txt')
            return float(blob.download_as_string())
        fs = gcsfs.GCSFileSystem()
        with fs.open(f'gs://models_detection/{model}.h5', 'rb') as model_file:
            model_gcs = h5py.File(model_file, 'r')
            return load_model(model_gcs)
    blob = MODELS_BUCKET.get_blob(model + '.joblib')
    return load(io.BytesIO(blob.download_as_bytes()))