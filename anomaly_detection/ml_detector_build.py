import os
import random
import datetime
import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from utils.config import (MODELS_PATH, PREDICTIONS_PATH, SEED, VERBOSITY,
                          FEATURES_NO_TIME_AND_COMMANDS, ENGINEERED_FEATURES,
                          PRESSURE_TEMPERATURE_FEATURES,
                          FEATURES_FOR_ANOMALY_DETECTION)
from utils.readers import DataReader, Preprocessor
from utils.plotters import Plotter


def get_preprocessed_data(raw=False,
                          features_to_read=FEATURES_FOR_ANOMALY_DETECTION):
    df = DataReader.load_data(raw=raw, features_to_read=features_to_read)
    df = Preprocessor.remove_step_zero(df)
    df = Preprocessor.feature_engineering(df)
    return df


def get_model(model):
    if 'IsolationForest' in model:
        detector = IsolationForest(n_estimators=1000,
                                   contamination=0.01,
                                   bootstrap=False,
                                   warm_start=True,
                                   random_state=SEED,
                                   verbose=VERBOSITY,
                                   n_jobs=-1)
    elif 'DBSCAN' in model:
        detector = DBSCAN(eps=0.1,
                          min_samples=10,
                          metric='euclidean',
                          algorithm='auto',
                          leaf_size=30,
                          p=None,
                          n_jobs=-1)
    elif 'LocalOutlierFactor' in model:
        detector = LocalOutlierFactor(n_neighbors=10,
                                      algorithm='auto',
                                      leaf_size=10,
                                      metric='minkowski',
                                      p=2,
                                      contamination=0.01,
                                      novelty=True,
                                      n_jobs=-1)
    return detector


def save_model_and_its_predictions(model,
                                   detector,
                                   detector_predict,
                                   timestamped=True,
                                   save_predictions=False):
    print(f'Saving the model')
    if timestamped:
        dump(
            detector,
            os.path.join(
                MODELS_PATH, 'anomaly_detectors',
                f'{model}_{datetime.datetime.now():%d%m_%H%M}.joblib'))
    else:
        dump(detector,
             os.path.join(MODELS_PATH, 'anomaly_detectors', f'{model}.joblib'))
    print(f'Model saving finished')
    if save_predictions:
        print('Saving predictions')
        if timestamped:
            pd.Series(detector_predict).astype(np.int8).to_csv(os.path.join(
                PREDICTIONS_PATH,
                f'{model}_{datetime.datetime.now():%d%m_%H%M}.csv'),
                                                               index=False)
        else:
            pd.Series(detector_predict).astype(np.int8).to_csv(os.path.join(
                PREDICTIONS_PATH, f'{model}.csv'),
                                                               index=False)
        print('Predictions saved')


if __name__ == '__main__':
    # os.chdir('..')
    df = get_preprocessed_data(raw=False)
    models = []
    trained_detectors = []
    for feature in ENGINEERED_FEATURES + PRESSURE_TEMPERATURE_FEATURES:
        model = f'LocalOutlierFactor_{feature}'
        models.append(model)
        detector = get_model(model)
        print(f'Fitting {model} started')
        if 'LocalOutlierFactor' in model:
            detector.fit(np.array(df[feature]).reshape(-1, 1))
            detector_predict = detector.predict(
                np.array(df[feature]).reshape(-1, 1))
        else:
            detector_predict = detector.fit_predict(
                np.array(df).reshape(-1, 1))
        print(f'Fitting {model} finished')
        trained_detectors.append(detector)
        df[f'ANOMALY_{feature}'] = pd.Series(detector_predict).astype(np.int8)
        save_model_and_its_predictions(model,
                                       detector,
                                       detector_predict,
                                       timestamped=False,
                                       save_predictions=False)