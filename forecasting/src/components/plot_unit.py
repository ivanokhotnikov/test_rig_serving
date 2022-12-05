import plotly.graph_objects as go


def plot_unit(df, feature):
    """
    The plot_unit function takes a dataframe and a feature as input, and returns an interactive plot of the time series for that feature.
    
    Args:
        df: Pass the dataframe that contains the feature
        feature: Select the feature to be plotted
    
    Returns:
        A plotly figure
    """
    fig = go.Figure()
    fig.add_scatter(x=df['TIME'], y=df[feature])
    fig.update_layout(template='none',
                      xaxis=dict(title='TEST TIME'),
                      yaxis_title=feature.replace('_', ' '))
    return fig
