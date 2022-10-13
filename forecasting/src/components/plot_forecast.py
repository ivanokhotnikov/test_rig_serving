from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_forecast(
    historical: pd.DataFrame,
    forecast: np.ndarray,
    feature: str,
    rolling_window: Optional[int] = None,
    new: Optional[pd.DataFrame] = None,
    plot_ma_all: Optional[bool] = False,
    plot_each_unit: Optional[bool] = False,
) -> go.Figure:
    fig = go.Figure()
    if plot_each_unit:
        for unit in historical['UNIT'].unique():
            for test in historical[historical['UNIT'] ==
                                   unit]['TEST'].unique():
                fig.add_scatter(
                    x=np.array(
                        historical[(historical['UNIT'] == unit)
                                   & (historical['TEST'] == test)].index) /
                    3600,
                    y=historical[(historical['UNIT'] == unit)
                                 & (historical['TEST'] == test)]
                    [feature].values.reshape(-1),
                    line=dict(width=1, ),
                    opacity=0.5,
                    name=f'{unit}-{test}',
                    showlegend=False,
                )
    else:
        fig.add_scatter(
            x=np.arange(len(historical)) / 3600,
            y=historical[feature].values.reshape(-1),
            line=dict(
                width=1,
                color='gray',
            ),
            opacity=0.5,
            name='Historical',
            showlegend=True,
        )
    if new is None:
        fig.add_scatter(
            x=np.arange(len(historical),
                        len(historical) + len(forecast) + 1) / 3600,
            y=forecast.reshape(-1),
            name='Forecast',
            line=dict(
                color='indianred',
                width=1.25,
            ),
        )
        if plot_ma_all and rolling_window:
            fig.add_scatter(
                x=np.arange(len(historical) + len(forecast) + 1) / 3600,
                y=pd.Series(
                    np.concatenate(
                        (historical[feature].values, forecast.reshape(-1)
                         ), )).rolling(rolling_window).mean().values,
                name='Moving average trend',
                line=dict(
                    color='orange',
                    width=1.5,
                ),
            )
    else:
        fig.add_scatter(
            x=np.arange(len(historical),
                        len(historical) + len(new) + 1) / 3600,
            y=new[feature].values.reshape(-1),
            name='New data',
            line=dict(
                color='steelblue',
                width=1.25,
            ),
        )
        fig.add_scatter(
            x=np.arange(
                len(historical) + len(new),
                len(historical) + len(new) + len(forecast) + 1) / 3600,
            y=forecast.reshape(-1),
            name='Forecast',
            line=dict(
                color='indianred',
                width=1.25,
            ),
        )
        if plot_ma_all and rolling_window:
            fig.add_scatter(
                x=np.arange(len(historical) + len(new) + len(forecast) + 1) /
                3600,
                y=pd.Series(
                    np.concatenate(
                        historical[feature].values,
                        new[feature].values,
                        forecast.reshape(-1),
                    )).rolling(rolling_window).mean().values,
                name='Moving average trend',
                line=dict(
                    color='orange',
                    width=1.5,
                ),
            )
    fig.update_layout(template='none',
                      xaxis=dict(title='Total running time, hours'),
                      yaxis_title=f'{feature}_kW' if 'POWER' in feature else f'{feature}',
                      title=f'{feature}_FORECAST',
                      legend=dict(orientation='h',
                                  yanchor='bottom',
                                  xanchor='right',
                                  x=1,
                                  y=1.01))
    return fig
