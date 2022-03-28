import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.config import FEATURES_FOR_FORECASTING
from utils.readers import get_preprocessed_data


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        a = dataset[i - look_back:i, 0]
        X.append(a)
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    df = get_preprocessed_data(raw=False,
                               features_to_read=FEATURES_FOR_FORECASTING)
    N_SPLITS = 3
    X = df['DURATION']
    y = df['Vibration 1']
    folds = TimeSeriesSplit(n_splits=N_SPLITS)
    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size
    univariate_df = df[['DURATION', 'Vibration 1']].copy()
    univariate_df.columns = ['ds', 'y']
    train = univariate_df.iloc[:train_size, :]
    x_train = pd.DataFrame(univariate_df.iloc[:train_size, 0])
    y_train = pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0])
    y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 1])
    data = univariate_df.filter(['y'])
    #Convert the dataframe to a numpy array
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(-1, 0))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data[:10]
    # Defines the rolling window
    look_back = 120
    # Split into train and test sets
    train, test = scaled_data[:train_size -
                              look_back, :], scaled_data[train_size -
                                                         look_back:, :]

    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    print(len(x_train), len(x_test))

    model = Sequential()
    model.add(
        LSTM(128,
             return_sequences=True,
             input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train,
              y_train,
              batch_size=1,
              epochs=5,
              validation_data=(x_test, y_test))
    model.summary()

    # Lets predict with the model
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])

    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])

    # Get the root mean squared error (RMSE) and MAE
    score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
    score_mae = mean_absolute_error(y_test[0], test_predict[:, 0])
    print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))

    x_train_ticks = univariate_df.head(train_size)['ds']
    y_train = univariate_df.head(train_size)['y']
    x_test_ticks = univariate_df.tail(test_size)['ds']

    # Plot the forecast
    f, ax = plt.subplots(1)
    f.set_figheight(6)
    f.set_figwidth(15)

    sns.lineplot(x=x_train_ticks, y=y_train, ax=ax, label='Train Set')
    sns.lineplot(x=x_test_ticks,
                 y=test_predict[:, 0],
                 ax=ax,
                 color='green',
                 label='Prediction')
    sns.lineplot(x=x_test_ticks,
                 y=y_test[0],
                 ax=ax,
                 color='orange',
                 label='Ground truth')

    ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}',
                 fontsize=14)
    ax.set_xlabel(xlabel='Date', fontsize=14)
    ax.set_ylabel(ylabel='Vibration 1', fontsize=14)

    plt.show()