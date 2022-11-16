import gcsfs
import h5py
from keras.models import load_model
from streamlit import cache


@cache(allow_output_mutation=True)
def import_forecaster(file_name):
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'gs://models_forecasting/{file_name}.h5',
                 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        keras_model = load_model(model_gcs)
    return keras_model