from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from components.constants import LOOKBACK


def plot_forecast(
    history_df: pd.DataFrame,
    forecast: np.ndarray,
    feature: str,
    new_forecast: Optional[np.ndarray] = None,
    rolling_window: Optional[int] = None,
    new_data_df: Optional[pd.DataFrame] = None,
    plot_each_unit: Optional[bool] = False,
) -> go.Figure:
    fig = go.Figure()
    if plot_each_unit:
        for unit in history_df['UNIT'].unique():
            for test in history_df[history_df['UNIT'] ==
                                   unit]['TEST'].unique():
                fig.add_scatter(
                    x=np.array(
                        history_df[(history_df['UNIT'] == unit)
                                   & (history_df['TEST'] == test)].index) /
                    3600,
                    y=history_df[(history_df['UNIT'] == unit)
                                 & (history_df['TEST'] == test)]
                    [feature].values.reshape(-1),
                    line=dict(width=1, ),
                    opacity=0.5,
                    name=f'{unit}-{test}',
                    showlegend=False,
                )
    else:
        fig.add_scatter(
            x=np.arange(len(history_df)) / 3600,
            y=history_df[feature].values.reshape(-1),
            line=dict(
                width=1,
                color='gray',
            ),
            opacity=0.5,
            name='History',
            showlegend=True,
        )
    if new_data_df is None:
        fig.add_scatter(
            x=np.arange(len(history_df),
                        len(history_df) + len(forecast) + 1) / 3600,
            y=forecast.reshape(-1),
            name='Forecast',
            line=dict(
                color='indianred',
                width=1,
            ),
        )
        if rolling_window:
            fig.add_scatter(
                x=np.arange(len(history_df) + len(forecast) + 1) / 3600,
                y=pd.Series(
                    np.concatenate(
                        (history_df[feature].values, forecast.reshape(-1)
                         ), )).rolling(rolling_window).mean().values,
                name='Moving average trend',
                line=dict(
                    color='orange',
                    width=1.25,
                ),
            )
    else:
        if new_data_df is not None:
            fig.add_scatter(
                x=np.arange(len(history_df),
                            len(history_df) + len(new_data_df) + 1) / 3600,
                y=new_data_df[feature].values.reshape(-1),
                name='New data',
                line=dict(
                    color='steelblue',
                    width=1,
                ),
            )
            if new_forecast is not None:
                fig.add_scatter(
                    x=np.arange(
                        len(history_df) + len(new_data_df),
                        len(history_df) + len(new_data_df) +
                        len(new_forecast) + 1) / 3600,
                    y=forecast.reshape(-1),
                    name='New forecast',
                    line=dict(
                        color='seagreen',
                        width=1,
                    ),
                )
        if forecast is not None:
            fig.add_scatter(
                x=np.arange(
                    len(history_df) + LOOKBACK,
                    LOOKBACK + len(history_df) + len(forecast) + 1) / 3600,
                y=forecast.reshape(-1),
                name='Old forecast',
                line=dict(
                    color='indianred',
                    width=1,
                ),
            )
            
        if rolling_window is not None and new_data_df is not None and new_forecast is not None:
            fig.add_scatter(
                x=np.arange(
                    len(history_df) + len(new_data_df) + len(new_forecast) + 1)
                / 3600,
                y=pd.Series(
                    np.concatenate(
                        (history_df[feature].values,
                         new_data_df[feature].values, new_forecast.reshape(-1)
                         ))).rolling(rolling_window).mean().values,
                name='Moving average trend',
                line=dict(
                    color='orange',
                    width=1.25,
                ),
            )
    fig.update_layout(
        template='none',
        xaxis=dict(title='Total running time, hours'),
        yaxis_title=f'{feature.lower().capitalize().replace("_", " ")}, kW'
        if 'POWER' in feature else
        f'{feature.lower().capitalize().replace("_", " ")}',
        title=f'{feature.lower().capitalize().replace("_", " ")} forecast',
        legend=dict(orientation='h',
                    yanchor='bottom',
                    xanchor='right',
                    x=1,
                    y=1.01))
    return fig
