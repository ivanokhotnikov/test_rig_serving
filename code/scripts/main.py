from utils.config import MODELS_PATH, IMAGES_PATH
from utils.readers import DataReader, Preprocessor, ModelReader
from utils.plotters import Plotter


def main():
    data = DataReader.load_data()


if __name__ == '__main__':
    main()