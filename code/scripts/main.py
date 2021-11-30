import os
from utils.config import MODELS_PATH, IMAGES_PATH, PREDICTIONS_PATH
from utils.readers import DataReader, Preprocessor, ModelReader
from utils.plotters import Plotter

from joblib import dump, load


if __name__ == '__main__':
    os.chdir('..\\..')
    dtc = load('outputs\\models\\iforest_2911_0359.joblib')