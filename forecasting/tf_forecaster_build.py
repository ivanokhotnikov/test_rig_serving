import datetime
import os

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from utils.config import (ENGINEERED_FEATURES, FORECAST_FEATURES,
                          LOCAL_DATA_PATH, PRESSURE_TEMPERATURE, VIBRATIONS)
from utils.readers import get_processed_data

if __name__ == '__main__':

    class WindowGenerator():

        def __init__(self,
                     input_width,
                     label_width,
                     shift,
                     train_df=None,
                     val_df=None,
                     test_df=None,
                     label_columns=None):
            # Store the raw data.
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df

            # Work out the label column indices.
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {
                    name: i
                    for i, name in enumerate(label_columns)
                }
            self.column_indices = {
                name: i
                for i, name in enumerate(train_df.columns)
            }

            # Work out the window parameters.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(
                self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(
                self.total_window_size)[self.labels_slice]

        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'
            ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([
                labels[:, :, self.column_indices[name]]
                for name in self.label_columns
            ],
                              axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    local = True
    if local:
        df = get_processed_data(local=True, features_to_read=FORECAST_FEATURES)
    else:
        df = get_processed_data(raw=False, features_to_read=FORECAST_FEATURES)

    window = 3600
    column_indices = {name: i for i, name in enumerate(FORECAST_FEATURES)}
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)