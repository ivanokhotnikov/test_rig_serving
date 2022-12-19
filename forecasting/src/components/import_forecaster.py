import gcsfs
import h5py
from keras.models import load_model

def import_forecaster(file_name):
    """
    The import_forecaster function imports a saved model from the GCS bucket. The function takes one argument, which is the name of the file to be imported. It returns a Keras model.
    
    Args:
        file_name: Specify the name of the model file
    
    Returns:
        A keras model
    """
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'gs://models_forecasting/{file_name}.h5',
                 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        keras_model = load_model(model_gcs)
    return keras_model