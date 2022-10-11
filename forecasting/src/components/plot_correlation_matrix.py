import plotly.graph_objects as go


def plot_correlation_matrix(df, features):
    return go.Figure(
        go.Heatmap(x=features,
                   y=features,
                   z=df[features].corr(method='pearson').values,
                   colorscale='inferno'))