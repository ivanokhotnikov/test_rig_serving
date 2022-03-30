import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from utils.config import (ENGINEERED_FEATURES, FEATURES_FOR_FORECASTING,
                          PRESSURE_TEMPERATURE, VIBRATIONS, LOCAL_DATA_PATH)
from utils.readers import get_preprocessed_data


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


if __name__ == '__main__':
    local = True
    if local:
        df = get_preprocessed_data(local=True,
                                   features_to_read=FEATURES_FOR_FORECASTING)
    else:
        df = get_preprocessed_data(raw=False,
                               features_to_read=FEATURES_FOR_FORECASTING)
    features = ENGINEERED_FEATURES + PRESSURE_TEMPERATURE + VIBRATIONS
    forecast_df = df[features + ['RUNNING HOURS']].copy()
    window = 3600
    ma_forecast_df = forecast_df[features].rolling(window).mean()
    train_size = int(.8 * len(forecast_df))
    train_df = forecast_df[:train_size]
    test_df = forecast_df[train_size:]
    tscv = TimeSeriesSplit(n_splits=5)
