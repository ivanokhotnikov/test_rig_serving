import os

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import dump, load
from colorama import Fore
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from utils.config import FEATURES_FOR_FORECASTING, ENGINEERED_FEATURES, PRESSURE_TEMPERATURE, VIBRATIONS, MODELS_PATH
from utils.readers import get_preprocessed_data

if __name__ == '__main__':
    df = get_preprocessed_data(raw=False,
                               features_to_read=FEATURES_FOR_FORECASTING)
    plot_all = False
    plot_only_trends = True
    save = False
    methods = ['seasonal_decompose']
    features = ENGINEERED_FEATURES + PRESSURE_TEMPERATURE + VIBRATIONS
    for method in methods:
        start_all = time.time()
        for feature in features:
            print(Fore.GREEN + f'Decomposing {feature} with {method}')
            start = time.time()
            if method == 'seasonal_decompose':
                decomposition = seasonal_decompose(
                    df[feature],
                    model='additive',
                    period=3600,
                )
            if method == 'stl':
                decomposition = STL(
                    df[feature],
                    period=3600,
                ).fit()
            end = time.time()
            print(Fore.BLUE + f'Decomposition took {end - start:.2f} seconds')
            if save:
                start = time.time()
                print(Fore.GREEN + f'Saving decomposition for {feature}')
                dump(
                    decomposition,
                    os.path.join(MODELS_PATH,
                                f'{feature}_{method}_decomposer.joblib'))
                end = time.time()
                print(Fore.BLUE + f'Saving took {end - start:.2f} seconds')
            if plot_all:
                for plot in ('observed', 'trend', 'seasonal', 'resid'):
                    fig = go.Figure(
                        go.Scatter(x=df['DURATION'],
                                   y=getattr(decomposition, plot)))
                    fig.update_layout(title=f'{plot.capitalize()} {feature}',
                                      xaxis_title='DURATION',
                                      template='none')
                    fig.show()
            if plot_only_trends:
                fig = go.Figure()
                fig.add_scatter(x=df['DURATION'],
                                y=df[feature],
                                name='observed',
                                line=dict(color='gray', width=.5))
                fig.add_scatter(x=df['DURATION'],
                                y=getattr(decomposition, '_trend'),
                                name='stats_trend')
                fig.add_scatter(x=df['DURATION'],
                                y=df[feature].rolling(3600).mean(),
                                name='pandas ma')
                fig.update_layout(yaxis_title=feature,
                                  xaxis_title='DURATION',
                                  template='none')
                fig.show()
        end_all = time.time()
        print(
            Fore.BLUE +
            f'Total time for {method} decomposition and saving: {end_all - start_all:.2f} seconds'
        )
