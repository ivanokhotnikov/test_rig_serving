import numpy as np
import plotly.graph_objects as go
from components.constants import LOOKBACK


def plot_forecast(history_df,
                  forecast,
                  feature,
                  new_forecast=None,
                  rolling_window=None,
                  new_data_df=None,
                  plot_each_unit=False) -> go.Figure:
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
                    [feature].values.flatten(),
                    line=dict(width=1.5, ),
                    opacity=0.5,
                    name=f'{unit}-{test}',
                    showlegend=False,
                )
    else:
        fig.add_scatter(
            x=np.arange(len(history_df)) / 3600,
            y=history_df[feature].values.flatten(),
            line=dict(
                width=1.5,
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
            y=forecast.flatten(),
            name='Forecast',
            line=dict(
                color='indianred',
                width=1.5,
            ),
        )
        if rolling_window:
            fig.add_scatter(
                x=np.arange(len(history_df) + len(forecast) + 1) / 3600,
                y=np.convolve(
                    np.concatenate((
                        history_df[feature].values,
                        forecast.flatten(),
                    )),
                    np.ones(rolling_window) / rolling_window,
                    'same',
                ),
                name='Moving average trend',
                line=dict(
                    color='orange',
                    width=1.75,
                ),
            )
    else:
        fig.add_scatter(
            x=np.arange(len(history_df),
                        len(history_df) + len(new_data_df) + 1) / 3600,
            y=new_data_df[feature].values.flatten(),
            name='New data',
            line=dict(
                color='steelblue',
                width=1.5,
            ),
        )
        if new_forecast is not None:
            fig.add_scatter(
                x=np.arange(
                    len(history_df) + len(new_data_df),
                    len(history_df) + len(new_data_df) + len(new_forecast) + 1)
                / 3600,
                y=new_forecast.flatten(),
                name='New forecast',
                line=dict(
                    color='seagreen',
                    width=1.5,
                ),
            )
            if rolling_window is not None:
                fig.add_scatter(
                    x=np.arange(
                        len(history_df) + len(new_data_df) +
                        len(new_forecast) + 1) / 3600,
                    y=np.convolve(
                        np.concatenate((
                            history_df[feature].values,
                            new_data_df[feature].values,
                            new_forecast.flatten(),
                        )),
                        np.ones(rolling_window) / rolling_window,
                        'same',
                    ),
                    name='Moving average trend',
                    line=dict(
                        color='orange',
                        width=1.75,
                    ),
                )
        fig.add_scatter(
            x=np.arange(
                len(history_df) + LOOKBACK,
                LOOKBACK + len(history_df) + len(forecast) + 1) / 3600,
            y=forecast.flatten(),
            name='Old forecast',
            line=dict(
                color='indianred',
                width=1.5,
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
                    y=1.01),
    )
    return fig
