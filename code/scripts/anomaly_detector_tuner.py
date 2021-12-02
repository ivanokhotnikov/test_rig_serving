import os
import optuna
import datetime

import pandas as pd
import numpy as np
from joblib import load

from utils.config import MODELS_PATH, PREDICTIONS_PATH, SEED, VERBOSITY, FEATURES_NO_TIME_AND_COMMANDS
from utils.readers import DataReader, Preprocessor
from utils.plotters import Plotter


if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('..\\..')
    print(os.getcwd())
    df = DataReader.load_data(raw=False)
    df = Preprocessor.remove_step_zero(df)
