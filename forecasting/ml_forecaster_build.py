import gc
import os

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from colorama import Fore
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras

from utils.config import (EARLY_STOPPING, MODELS_PATH, RAW_FORECAST_FEATURES,
                          VERBOSITY)
from utils.readers import get_ma_data, get_processed_data


def plot_acfs(df, features, window):
    for feature in features:
        fig = go.Figure()
        fig.add_scatter(y=df[feature],
                        name='observed',
                        line={
                            'color': 'gray',
                            'width': .25
                        })
        fig.add_scatter(y=df[feature].rolling(window).mean(),
                        name=f'ma window={window}')
        fig.update_layout(template='none', yaxis_title=feature)
        fig.show()


def plot_folds(train_df, tscv):
    for i, (train_idx, val_idx) in enumerate(tscv.split(train_df)):
        fig = go.Figure()
        fig.add_scatter(x=train_df.loc[train_idx, 'TIME'],
                        y=train_df.loc[train_idx, 'DRIVE POWER'],
                        line=dict(color='steelblue'),
                        name='Train')
        fig.add_scatter(x=train_df.loc[val_idx, 'TIME'],
                        y=train_df.loc[val_idx, 'DRIVE POWER'],
                        line=dict(color='indianred'),
                        name='Val')
        fig.update_layout(
            title=f'Fold {i+1}',
            xaxis_range=[train_df['TIME'].min(), train_df['TIME'].max()],
            template='none',
            legend=dict(orientation='h',
                        yanchor='bottom',
                        xanchor='right',
                        x=1,
                        y=1.01))
        fig.show()


def plot_predictions(train_df, test_df, feature, pred):
    fig = go.Figure()
    fig.add_scatter(x=train_df['TIME'], y=train_df[feature], name='train')
    fig.add_scatter(x=test_df['TIME'], y=test_df[feature], name='test')
    fig.add_scatter(x=test_df['TIME'], y=pred, name='pred')
    fig.update_layout(
        xaxis_range=[train_df['TIME'].min(), test_df['TIME'].max()],
        yaxis_title=feature,
        template='none',
        legend=dict(orientation='h',
                    yanchor='bottom',
                    xanchor='right',
                    x=1,
                    y=1.01))
    fig.show()


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, 'b.-', label='Training loss')
    plt.plot(np.arange(len(val_loss)) + 1,
             val_loss,
             'r.-',
             label='Validation loss')
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)


def get_model(algorithm):
    if algorithm == 'lgb':
        return lgb.LGBMRegressor(device='gpu',
                                 n_estimators=5000,
                                 learning_rate=0.001,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 max_depth=10,
                                 num_leaves=50,
                                 min_child_weight=300)
    if algorithm == 'xgb':
        return xgb.XGBRegressor(tree_method='gpu_hist',
                                gpu_id=-1,
                                objective='reg:squarederror',
                                n_estimators=5000,
                                learning_rate=0.001,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                max_depth=10)
    if algorithm == 'rnn':
        model = keras.models.Sequential([
            keras.layers.LSTM(20, return_sequences=True, input_shape=[None,
                                                                      1]),
            keras.layers.LSTM(20, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ])

        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model


if __name__ == '__main__':
    df = get_processed_data(raw=True,
                            local=False,
                            features_to_read=RAW_FORECAST_FEATURES)
    # df = get_ma_data()
    train_size = int(.5 * len(df))
    val_size = int(.25 * len(df))
    test_size = len(df) - train_size - val_size
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    tscv = TimeSeriesSplit(n_splits=5)
    feature = 'DRIVE POWER'
    algo = 'rnn'
    for i, (train_idx, val_idx) in enumerate(tscv.split(train_df), start=1):
        print(Fore.CYAN + f'Fold {i}' + Fore.RESET)
        X_train, y_train = train_df.loc[
            train_idx, 'TOTAL SECONDS'].values.reshape(
                -1, 1), train_df.loc[train_idx, feature].values.reshape(-1, 1)
        X_val, y_val = train_df.loc[val_idx, 'TOTAL SECONDS'].values.reshape(
            -1, 1), train_df.loc[val_idx, feature].values.reshape(-1, 1)
        model = get_model(algo)
        if algo == 'rnn':
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=128,
                verbose=VERBOSITY,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=EARLY_STOPPING,
                                                  monitor='val_loss',
                                                  mode='min',
                                                  verbose=VERBOSITY,
                                                  restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.75,
                        patience=EARLY_STOPPING // 2,
                        verbose=VERBOSITY,
                        mode='min'),
                    keras.callbacks.ModelCheckpoint('rnn.keras',
                                                    save_best_only=True)
                ])
        else:
            model.fit(X_train,
                      y_train,
                      eval_set=[(X_train, y_train), (X_val, y_val)],
                      early_stopping_rounds=50,
                      verbose=100)
        val_pred = model.predict(X_val)
        print(
            Fore.CYAN +
            f'Fold {i} RMSE: {mean_squared_error(y_val, val_pred, squared=False)}'
            + Fore.RESET)
        if algo == 'rnn':
            pass
        else:
            dump(model, os.path.join(MODELS_PATH, f'{algo}_ma_{i}.joblib'))
        if i != tscv.n_splits:
            del model
        del X_train, y_train, X_val, y_val
        gc.collect()
    # model = load(os.path.join(MODELS_PATH, f'{algo}_ma_5.joblib'))
    plot_predictions(
        train_df, test_df, feature,
        model.predict(test_df['TOTAL SECONDS'].values.reshape(-1, 1)))
