import os
import random
import datetime
import numpy as np
import pandas as pd
from pickle import dump, load

from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from utils.config import MODELS_PATH, PREDICTIONS_PATH, SEED, VERBOSITY, FEATURES_NO_TIME_AND_COMMANDS
from utils.readers import DataReader, Preprocessor
from utils.plotters import Plotter


def get_preprocessed_data(raw=False):
    df = DataReader.load_data(raw=raw)
    df = Preprocessor.remove_step_zero(df)
    return df


def get_model(model):
    if 'IsolationForest' in model:
        detector = IsolationForest(n_estimators=5000,
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
        detector = LocalOutlierFactor(n_neighbors=20,
                                      algorithm='auto',
                                      leaf_size=30,
                                      metric='minkowski',
                                      p=2,
                                      contamination=0.01,
                                      novelty=False,
                                      n_jobs=-1)
    return detector


def save_model_and_its_predictions(model,
                                   detector,
                                   detector_predict,
                                   timestamped=True):
    print(f'Saving the model')
    if timestamped:
        dump(
            detector,
            os.path.join(
                MODELS_PATH,
                f'{model}_{datetime.datetime.now():%d%m_%H%M}.joblib'))
    else:
        dump(detector, os.path.join(MODELS_PATH, f'{model}.joblib'))
    print(f'Model saving finished')
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
    print('Saving predictions finished')


def fit_predict_model(df, model, detector):
    print(f'Fitting {model} started')
    detector_predict = detector.fit_predict(np.array(df).reshape(-1, 1))
    print(f'Fitting finished')
    return detector, detector_predict


if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('..\\..')
    print(os.getcwd())
    df = get_preprocessed_data(raw=False)
    for feature in FEATURES_NO_TIME_AND_COMMANDS:
        Plotter.plot_all_per_step_feature(df, step=23, feature=feature)
    models = []
    trained_detectors = []
    for feature in FEATURES_NO_TIME_AND_COMMANDS:
        model = f'IsolationForest_{feature}'
        models.append(model)
        detector = get_model(model)
        detector, detector_predict = fit_predict_model(df[feature], model,
                                                       detector)
        trained_detectors.append(detector)
        df[f'ANOMALY_{feature}'] = pd.Series(detector_predict).astype(np.int8)
        save_model_and_its_predictions(model,
                                       detector,
                                       detector_predict,
                                       timestamped=False)
    anomalous_units = df[df['ANOMALY'] == -1]['UNIT'].unique()
    normal_units = [
        unit for unit in df['UNIT'].unique() if unit not in anomalous_units
    ]
    most_anomalous_units = df[df['ANOMALY'] ==
                              -1]['UNIT'].value_counts().index[:5]
    for unit in most_anomalous_units:
        Plotter.plot_anomalies_per_unit_feature(df,
                                                unit=unit,
                                                feature='M4 ANGLE')
    normal_unit = random.choice(normal_units)
    for feature in FEATURES_NO_TIME_AND_COMMANDS:
        Plotter.plot_anomalies_per_unit_feature(df,
                                                unit=most_anomalous_units[0],
                                                feature=feature)
        Plotter.plot_unit_per_feature(df, unit=normal_unit, feature=feature)