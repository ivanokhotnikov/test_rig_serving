import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import dump, load
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from utils.config import FEATURES_FOR_FORECASTING, ENGINEERED_FEATURES, PRESSURE_TEMPERATURE, VIBRATIONS, MODELS_PATH
from utils.readers import get_preprocessed_data

if __name__ == '__main__':
    df = get_preprocessed_data(raw=False,
                               features_to_read=FEATURES_FOR_FORECASTING)
    plot_all = False
    plot_only_trends = True
    methods = ('seasonal_decompose', 'stl')
    features = ENGINEERED_FEATURES + PRESSURE_TEMPERATURE + VIBRATIONS
    for method in methods:
        for feature in features:
            print(f'Decomposing {feature} with {method}')
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
            print(f'Saving decomposition for {feature}')
            dump(
                decomposition,
                os.path.join(MODELS_PATH,
                             f'{feature}_{method}_decomposer.joblib'))
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
                                y=getattr(decomposition, 'trend'),
                                name='trend')
                fig.update_layout(yaxis_title=feature,
                                  xaxis_title='DURATION',
                                  template='none')
                fig.show()