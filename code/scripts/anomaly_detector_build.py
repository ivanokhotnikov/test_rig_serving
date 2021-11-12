import os
import random
import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.ensemble import IsolationForest

from utils.config import MODELS_PATH, SEED, VERBOSITY, FEATURES_NO_TIME_AND_COMMANDS
from utils.readers import DataReader, Preprocessor
from utils.plotters import Plotter

if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('..\\..')
    print(os.getcwd())
    df = DataReader.load_data(raw=False)
    df = Preprocessor.remove_step_zero(df)
    train = False
    if train:
        iforest = IsolationForest(n_estimators=10000,
                                  contamination=0.01,
                                  bootstrap=True,
                                  warm_start=True,
                                  random_state=SEED,
                                  verbose=VERBOSITY,
                                  n_jobs=-1)
        iforest_predict = iforest.fit_predict(
            df[FEATURES_NO_TIME_AND_COMMANDS])
        dump(iforest, os.path.join(MODELS_PATH, 'iforest.joblib'))
        df['ANOMALY'] = pd.Series(iforest_predict).astype(np.int8)
    else:
        df['ANOMALY'] = pd.read_csv(
            os.path.join('outputs', 'predictions', 'iforest.csv'))
    anomalous_units = df[df['ANOMALY'] == -1]['UNIT'].unique()
    normal_units = [
        unit for unit in df['UNIT'].unique() if unit not in anomalous_units
    ]
    anomlaies_per_unit = df[df['ANOMALY'] == -1]['UNIT'].value_counts()
    most_anomalous_units = anomlaies_per_unit.index[:5]
    for unit in most_anomalous_units:
        Plotter.plot_anomalies_per_unit_feature(df, unit=unit, feature='PT4')
    normal_unit = random.choice(normal_units)
    for feature in FEATURES_NO_TIME_AND_COMMANDS:
        Plotter.plot_anomalies_per_unit_feature(df,
                                                unit=most_anomalous_units[0],
                                                feature=feature)
        Plotter.plot_unit_per_feature(df, unit=normal_unit, feature=feature)