import plotly.graph_objects as go
import logging
import streamlit as st


def plot_anomalies(
    df,
    unit=89,
    test=1,
    feature='M4 ANGLE',
):
    if not any(df[(df['UNIT'] == unit)
                  & (df['TEST'] == test)][f'ANOMALY_{feature.replace(" ","_")}'] == -1):
        st.write(f'No anomalies in the {feature} data')
        logging.info(f'No anomalies in the {feature} data')
        return None
    fig = go.Figure()
    fig.add_scatter(
        x=df[(df['UNIT'] == unit)
             & (df['TEST'] == test)]['TIME'],
        y=df[(df['UNIT'] == unit)
             & (df['TEST'] == test)][feature.replace(" ","_")],
        mode='lines',
        name='Data',
        showlegend=False,
        line={'color': 'steelblue'},
    )
    fig.add_scatter(
        x=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
             (df[f'ANOMALY_{feature.replace(" ","_")}'] == -1)]['TIME'],
        y=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
             (df[f'ANOMALY_{feature.replace(" ","_")}'] == -1)][feature.replace(" ","_")],
        mode='markers',
        name='Anomaly',
        line={'color': 'indianred'},
    )
    fig.update_layout(yaxis={'title': feature},
                      template='none',
                      title=f'Unit {unit}-{test}, {feature}',
                      legend=dict(orientation='h',
                                  yanchor='bottom',
                                  xanchor='right',
                                  x=1,
                                  y=1.01))
    return fig
