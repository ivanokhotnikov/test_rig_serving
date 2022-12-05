import plotly.graph_objects as go
import logging
import streamlit as st


def plot_anomalies(df, unit=89, test=1, feature='M4 ANGLE'):
    """
    The plot_anomalies function plots the data and anomalies for a given unit, test, and feature.
    
    Args:
        df: Pass the dataframe to the function
        unit=89: Select the unit number from the dataframe
        test=1: Select the data from a specific test
        feature='M4 ANGLE': Select the feature to plot
    
    Returns:
        A plotly figure with the data plotted
    """
    if not any(df[(df['UNIT'] == unit)
                  & (df['TEST'] == test)]
               [f'ANOMALY_{feature.replace(" ","_")}'] == -1):
        st.write(f'No anomalies in the {feature} data')
        logging.info(f'No anomalies in the {feature} data')
        return None
    fig = go.Figure()
    fig.add_scatter(x=df[(df['UNIT'] == unit)
                         & (df['TEST'] == test)]['TIME'],
                    y=df[(df['UNIT'] == unit)
                         & (df['TEST'] == test)][feature.replace(" ", "_")],
                    mode='lines',
                    name='Data',
                    showlegend=False,
                    line={'color': 'steelblue'})
    fig.add_scatter(
        x=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
             (df[f'ANOMALY_{feature.replace(" ","_")}'] == -1)]['TIME'],
        y=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
             (df[f'ANOMALY_{feature.replace(" ","_")}']
              == -1)][feature.replace(" ", "_")],
        mode='markers',
        name='Anomaly',
        line={'color': 'indianred'})
    fig.update_layout(yaxis={'title': feature},
                      template='none',
                      title=f'Unit {unit}-{test}, {feature}',
                      legend=dict(orientation='h',
                                  yanchor='bottom',
                                  xanchor='right',
                                  x=1,
                                  y=1.01))
    return fig
