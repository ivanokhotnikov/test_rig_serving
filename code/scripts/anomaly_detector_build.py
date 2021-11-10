import os
from utils.config import MODELS_PATH, IMAGES_PATH
from utils.readers import DataReader, Preprocessor
from utils.plotters import Plotter

from joblib import dump, load


def main():
    print(os.getcwd())
    os.chdir('..\\..')
    print(os.getcwd())
    df = DataReader.load_data()


if __name__ == '__main__':
    main()