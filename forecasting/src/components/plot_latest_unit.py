import plotly.graph_objects as go

def plot_latest_unit(df, feature):
        fig = go.Figure()
        fig.add_scatter(
            x=df['TIME'],
            y=df[feature]
        )
        fig.update_layout(template='none',
                        xaxis=dict(title='Test time'),
                        yaxis_title=feature)
        return fig