from components import create_sequences
from components.constants import LOOKBACK
from streamlit import cache


@cache
def predict(data_df, feature, scaler, forecaster):
    scaled_data = scaler.transform(data_df[feature].values.reshape(-1, 1))
    sequenced_scaled_data = create_sequences(scaled_data,
                                             lookback=LOOKBACK,
                                             inference=True)
    unscaled_forecast = forecaster.predict(sequenced_scaled_data)
    forecast = scaler.inverse_transform(unscaled_forecast)
    return forecast
