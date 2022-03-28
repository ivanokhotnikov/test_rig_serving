from utils.config import (FEATURES_FOR_FORECASTING, ENGINEERED_FEATURES,
                          PRESSURE_TEMPERATURE_FEATURES)
from utils.readers import get_preprocessed_data

if __name__ == '__main__':
    df = get_preprocessed_data(raw=False,
                               features_to_read=FEATURES_FOR_FORECASTING)
    forecast_df = df[ENGINEERED_FEATURES + PRESSURE_TEMPERATURE_FEATURES +
                     ['Vibration 1', ' Vibration 2', 'RUNNING HOURS']].copy()
