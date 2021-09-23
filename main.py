import os
import pickle
import datetime
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss

import optuna

from xgboost import XGBClassifier

from utils import clock
import utils.readers as r
import utils.plotters as p
import utils.config as c


def load_data():
    if read_all:
        if raw:
            return r.read_all_raw_data()
        else:
            return r.read_combined_data()
    else:
        if raw:
            return r.read_raw_unit_data(unit_id=unit)
        else:
            return pd.DataFrame(r.read_summary_file())


def remove_outliers(df, z_score):
    return df[(np.abs(stats.zscore(df[c.FEATURES_NO_TIME])) < z_score).all(
        axis=1)]


def remove_step_zero(df):
    return df.drop(df[df['STEP'] == 0].index, axis=0)


def cv(params, df):
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(df[features].values,
                                                        df[target].values,
                                                        random_state=c.SEED,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        stratify=df[target])
    skf = StratifiedKFold(n_splits=c.FOLDS, shuffle=True)
    oof = np.zeros(X_TRAIN.shape)
    pred = np.zeros(X_TEST.shape)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_TRAIN, Y_TRAIN)):
        print(f'Fold {fold+1} started')
        x_train, x_val = X_TRAIN[train_idx], X_TRAIN[val_idx]
        y_train, y_val = Y_TRAIN[train_idx], Y_TRAIN[val_idx]
        xgb = XGBClassifier(**params)
        xgb.fit(x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric='merror',
                early_stopping_rounds=c.EARLY_STOPPING_ROUNDS,
                verbose=c.VERBOSITY)
        oof[val_idx] = xgb.predict_proba(x_val)
        pred += xgb.predict_proba(X_TEST) / c.FOLDS
    return log_loss(Y_TRAIN, oof), log_loss(Y_TEST, pred)


def objective(trial):
    x_train, x_val, y_train, y_val = train_test_split(
        X_TRAIN,
        Y_TRAIN,
        test_size=0.2,
        shuffle=True,
        stratify=Y_TRAIN,
    )
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 100),
        'reg_lambda': trial.suggest_int('reg_lambda', 1, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6,
                                                 0.9),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 0.9),
        'gamma': trial.suggest_int('gamma', 0, 20),
        'n_estimators': 1000,
        'eta': 0.1,
        'tree_method': 'gpu_hist',
        'use_label_encoder': False,
    }
    xgb = XGBClassifier(**params)
    xgb.fit(x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric='mlogloss',
            early_stopping_rounds=c.EARLY_STOPPING_ROUNDS,
            verbose=c.VERBOSITY)
    preds = xgb.predict_proba(x_val)
    return log_loss(y_val, preds)


if __name__ == '__main__':
    raw = False
    read_all = True
    optimize = True
    is_optimized = False
    unit = 'HYD000089-R1_RAW' if not read_all else None

    df = load_data()
    df = remove_outliers(df, z_score=5)
    target = 'STEP'
    features = c.FEATURES_NO_TIME_AND_COMMANDS
    features.remove(target)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    if optimize:
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
            df[features].values,
            df[target].values,
            test_size=0.2,
            stratify=df[target],
            random_state=c.SEED,
            shuffle=True)
        optimizer = optuna.create_study(direction='minimize',
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.MedianPruner())
        optimizer.optimize(objective, timeout=c.OPTIMIZATION_TIME_BUDGET)
        best_params = optimizer.best_params
        flag = 'opt'
    else:
        best_params = {
            'max_depth': 15,
            'reg_alpha': 6,
            'reg_lambda': 36,
            'min_child_weight': 18,
            'subsample': 0.8797985875169991,
            'colsample_bytree': 0.7440469154006446,
            'colsample_bylevel': 0.7381166426468402,
            'colsample_bynode': 0.6722527411125266,
            'gamma': 2,
            'n_estimators': 1000,
            'eta': 0.1,
            'tree_method': 'gpu_hist',
            'use_label_encoder': False,
        }
        flag = 'reg'
    oof = np.zeros(X_TRAIN.shape)
    pred = np.zeros(X_TEST.shape)
    skf = StratifiedKFold(n_splits=c.FOLDS, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_TRAIN, Y_TRAIN)):
        print(f'Fold {fold+1} started')
        x_train, x_val = X_TRAIN[train_idx], X_TRAIN[val_idx]
        y_train, y_val = Y_TRAIN[train_idx], Y_TRAIN[val_idx]
        clf = XGBClassifier(**best_params,
                            n_estimators=10000,
                            eta=0.01,
                            use_label_encoder=False,
                            tree_method='gpu_hist')
        clf.fit(x_train,
                y_train,
                eval_metric='mlogloss',
                eval_set=[(x_val, y_val)],
                early_stopping_rounds=c.EARLY_STOPPING_ROUNDS,
                verbose=c.VERBOSITY)
        oof[val_idx] = clf.predict_proba(x_val)
        pred += clf.predict_proba(X_TEST) / c.FOLDS
        print(f'Fold {fold+1} log_loss: {log_loss(y_val, oof[val_idx]):.4f}\n')
    val_score = log_loss(Y_TRAIN, oof)
    test_score = log_loss(Y_TEST, pred)
    clf.save_model(
        os.path.join(
            c.MODELS_PATH,
            f'{flag}_{val_score:.4f}_{test_score:.4f}_{datetime.datetime.now():%d%m_%I%M}.json'
        ))
