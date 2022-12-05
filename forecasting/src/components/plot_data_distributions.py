import json

import plotly.figure_factory as ff
from components.constants import MODELS_BUCKET


def plot_data_distributions(feature,
                            processed_df,
                            forecast=None,
                            new_data_df=None,
                            new_forecast=None):
    """
    The plot_data_distributions function plots the distribution of data for a given feature. It takes as input: the feature to plot, the processed_df DataFrame containing the training and test data for that feature.
    
    Args:
        feature: Select the feature to plot
        processed_df: A pandas dataframe with the processed data
        forecast=None: An array with the forecast
        new_data_df=None: A pandas dataframe with the new data
        new_forecast=None: An array with the new forecast
    
    Returns:
        A plotly figure
    """
    blob = MODELS_BUCKET.get_blob(f'{feature}_params.json')
    train_data_size = json.loads(blob.download_as_bytes())['train_data_size']
    train_data = processed_df.loc[:int(len(processed_df) * train_data_size),
                                  feature].values
    test_data = processed_df.loc[int(len(processed_df) * train_data_size):,
                                 feature].values
    forecast = forecast.flatten()
    data = [train_data, test_data, forecast]
    labels = [
        f'Train, {train_data_size*100:.1f}%',
        f'Test, {(1-train_data_size)*100:.1f}%', 'Forecast'
    ]
    if new_data_df is not None:
        new_data = new_data_df[feature].values.flatten()
        data.append(new_data)
        labels.append('New data')
    if new_forecast is not None:
        new_forecast = new_forecast.flatten()
        data.append(new_forecast)
        labels.append('New forecast')
    fig = ff.create_distplot(data, labels, show_hist=False, show_rug=False)
    fig.update_layout(
        template='none',
        title=f'{feature.capitalize().replace("_", " ")} data distributions',
        legend=dict(orientation='h',
                    yanchor='bottom',
                    xanchor='right',
                    x=1,
                    y=1.01))
    return fig