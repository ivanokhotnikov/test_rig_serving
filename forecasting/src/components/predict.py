from components import create_sequences
from components.constants import LOOKBACK
from streamlit import cache


@cache
def predict(data_df, feature, scaler, forecaster):
    """
    The predict function takes a dataframe containing the feature to be forecasted, the scaler used to normalize the data, and the forecaster model. It returns an array of forecasts for each time step in the input feature.
    
    Args:
        data_df: Pass the dataframe that contains the feature we want to forecast
        feature: Specify the column in the dataframe that contains the feature we want to predict
        scaler: Inverse the transformation of the forecast
        forecaster: Predict the values with the trained forecaster model
    
    Returns:
        A numpy array of the same length as the input data
    """
    scaled_data = scaler.transform(data_df[feature].values.reshape(-1, 1))
    sequenced_scaled_data = create_sequences(scaled_data,
                                             lookback=LOOKBACK,
                                             inference=True)
    unscaled_forecast = forecaster.predict(sequenced_scaled_data)
    forecast = scaler.inverse_transform(unscaled_forecast)
    return forecast