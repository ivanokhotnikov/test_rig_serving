import plotly.graph_objects as go


def plot_correlation_matrix(df, features):
    """
    The plot_correlation_matrix function creates a heatmap of the correlation matrix for the specified features. The function takes two parameters: df and features, which are expected to be a Pandas DataFrame and list, respectively.
    
    Args:
        df: Pass the dataframe that contains the features
        features: Define the features that should be included in the correlation matrix
    
    Returns:
        A plotly figure object
    """
    return go.Figure(
        go.Heatmap(x=features,
                   y=features,
                   z=df[features].corr(method='pearson').values,
                   showscale=False))
