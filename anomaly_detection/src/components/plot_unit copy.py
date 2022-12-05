import plotly.graph_objects as go


def plot_unit(df, feature):
    """
    The plot_unit function takes a dataframe and a feature as input, and returns an interactive plot of the time series for that feature.
    
    Args:
        df: Pass the dataframe that contains the time and feature values
        feature: Specify the feature that should be plotted
    
    Returns:
        A plotly figure
    """
    fig = go.Figure()
    fig.add_scatter(x=df['TIME'], y=df[feature])
    fig.update_layout(template='none',
                      xaxis=dict(title='Test time'),
                      yaxis_title=feature.lower().capitalize().replace(
                          '_', ' '))
    return fig
