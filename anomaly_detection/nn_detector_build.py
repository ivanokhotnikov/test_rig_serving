import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow import keras
from keras import layers

from utils.config import (MODELS_PATH, PREDICTIONS_PATH, SEED, VERBOSITY,
                          FEATURES_NO_TIME_AND_COMMANDS, ENGINEERED_FEATURES,
                          PRESSURE_TEMPERATURE_FEATURES,
                          FEATURES_FOR_ANOMALY_DETECTION, TIME_STEPS,
                          EARLY_STOPPING)
from utils.readers import DataReader, Preprocessor
from utils.plotters import Plotter


def get_preprocessed_data(raw=False,
                          features_to_read=FEATURES_FOR_ANOMALY_DETECTION):
    df = DataReader.load_data(raw=raw, features_to_read=features_to_read)
    df = Preprocessor.remove_step_zero(df)
    df = Preprocessor.feature_engineering(df)
    return df


def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i:(i + time_steps)])
    return np.stack(output)


if __name__ == '__main__':
    os.chdir('..')
    df = get_preprocessed_data(raw=False)
    test_lengths = []
    step_lengths = []
    for unit in df['UNIT'].unique():
        for unit_test in df[df['UNIT'] == unit]['TEST'].unique():
            test_lengths.append(
                len(df[(df['UNIT'] == unit) & (df['TEST'] == unit_test)]))
            for step in df[(df['UNIT'] == unit)
                           & (df['TEST'] == unit_test)]['STEP'].unique():
                step_lengths.append(
                    len(df[(df['UNIT'] == unit) & (df['TEST'] == unit_test) &
                           (df['STEP'] == step)]))
    mean_test_dur_sec = np.mean(test_lengths)
    mean_step_dur_sec = np.mean(step_lengths)
    print(
        f'Mean test duration {mean_test_dur_sec:.2f} seconds = {mean_test_dur_sec/60:.2f} minutes = {mean_test_dur_sec/3600:.2f} hours'
    )
    print(
        f'Mean step duration {mean_step_dur_sec:.2f} seconds = {mean_step_dur_sec/60:.2f} minutes = {mean_step_dur_sec/3600:.2f} hours'
    )
    trained_detectors = []
    for feature in ENGINEERED_FEATURES + PRESSURE_TEMPERATURE_FEATURES:
        model = f'ConvolutionalAutoencoder_{feature}'
        train_data, test_data = train_test_split(df[feature],
                                                 train_size=0.8,
                                                 shuffle=False)
        train_data.sort_index(inplace=True)
        test_data.sort_index(inplace=True)
        anomaly_df = pd.DataFrame(test_data[TIME_STEPS - 1:].copy())
        anomaly_df.index = test_data[TIME_STEPS - 1:].index

        scaler = StandardScaler()
        scaled_train_data = scaler.fit_transform(
            train_data.values.reshape(-1, 1))
        dump(
            scaler,
            os.path.join(MODELS_PATH, 'anomaly_detectors',
                         f'{model}_scaler.joblib'))

        x_train = create_sequences(scaled_train_data.reshape(-1, 1))
        print(x_train.shape)

        detector = keras.Sequential([
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(filters=32,
                          kernel_size=6,
                          padding='same',
                          strides=2,
                          activation='relu'),
            layers.Dropout(rate=0.25),
            layers.Conv1D(filters=16,
                          kernel_size=6,
                          padding='same',
                          strides=2,
                          activation='relu'),
            layers.Conv1DTranspose(filters=16,
                                   kernel_size=6,
                                   padding='same',
                                   strides=2,
                                   activation='relu'),
            layers.Dropout(rate=0.25),
            layers.Conv1DTranspose(filters=32,
                                   kernel_size=6,
                                   padding='same',
                                   strides=2,
                                   activation='relu'),
            layers.Conv1DTranspose(filters=1, kernel_size=6, padding='same'),
        ])
        detector.summary()

        detector.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        )

        history = detector.fit(
            x_train,
            x_train,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            verbose=VERBOSITY,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=EARLY_STOPPING,
                                              monitor='val_loss',
                                              mode='min',
                                              verbose=VERBOSITY,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.75,
                                                  patience=EARLY_STOPPING // 2,
                                                  verbose=VERBOSITY,
                                                  mode='min')
            ])

        trained_detectors.append(detector)
        detector.save(
            os.path.join(MODELS_PATH, 'anomaly_detectors', f'{model}.h5'))

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

        # Get train MAE loss.
        x_train_pred = detector.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

        plt.hist(train_mae_loss, bins=50)
        plt.xlabel('Train MAE loss')
        plt.ylabel('No of samples')
        plt.show()

        # Get reconstruction loss threshold.
        threshold = np.percentile(train_mae_loss, q=95)
        print('Reconstruction error threshold: ', threshold)
        with open(
                os.path.join(MODELS_PATH, 'anomaly_detectors',
                             f'{model}_threshold.txt'), 'w+') as f:
            f.write(str(threshold)),

        scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))

        # Create sequences from test values.
        x_test = create_sequences(scaled_test_data.reshape(-1, 1))
        print('Test input shape: ', x_test.shape)

        # Get test MAE loss.
        x_test_pred = detector.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))

        plt.hist(test_mae_loss, bins=50)
        plt.xlabel('Test MAE loss')
        plt.ylabel('No of samples')
        plt.show()

        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > threshold
        print('Number of anomaly samples: ', np.sum(anomalies))
        print('Indices of anomaly samples: ', np.where(anomalies))

        # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
        anomalous_data_indices = []
        for data_idx in range(TIME_STEPS - 1,
                              len(scaled_test_data) - TIME_STEPS + 1):
            if np.all(anomalies[data_idx - TIME_STEPS + 1:data_idx]):
                anomalous_data_indices.append(data_idx)

        df.loc[:,f'ANOMALY_{feature}'] = 1
        df.loc[anomalous_data_indices, f'ANOMALY_{feature}'] = -1

        plt.figure(figsize=(15, 5))
        plt.plot(test_data.index, test_data.values.reshape((-1)))
        plt.scatter(test_data.iloc[anomalous_data_indices].index,
                    test_data.iloc[anomalous_data_indices].values.reshape(
                        (-1)),
                    color='r')
        plt.ylabel(feature)
        plt.show()
