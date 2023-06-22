import numpy as np
import pytest
from components.build_power_features import build_power_features
from components.get_raw_data_files import get_raw_data_files
from components.get_raw_data_folder_stats import get_raw_data_folder_stats
from components.import_forecast_features import import_forecast_features
from components.import_forecaster import import_forecaster
from components.import_scaler import import_scaler
from components.plot_correlation_matrix import plot_correlation_matrix
from components.plot_forecast import plot_forecast
from components.plot_unit import plot_unit
from components.predict import predict
from components.read_latest_unit import read_latest_unit
from components.read_processed_data import read_processed_data
from components.read_unit_data import read_unit_data
from components.remove_step_zero import remove_step_zero

features = import_forecast_features()
df = read_processed_data(features=features + ['UNIT', 'TEST', 'TIME'])
_, files = get_raw_data_folder_stats()
random_unit_int = np.random.choice(files)
random_unit_file_name = get_raw_data_files(random_unit_int)[0]
new_data_df = read_unit_data(random_unit_file_name)
new_data_df_wo_zero = remove_step_zero(new_data_df)
new_interim_df = build_power_features(new_data_df_wo_zero)
latest_unit_df = read_latest_unit(df)
new_data_list = [new_interim_df, None]
window_list = [3200, None]
plot_each_unit_list = [True, False]

forecaster = import_forecaster(features[0])
scaler = import_scaler(features[0])


def test_plot_correlation_matrix(capsys):
    plot_correlation_matrix(df, features)
    out, err = capsys.readouterr()
    assert err == ''


def test_plot_unit(capsys):
    plot_unit(df, features[0])
    out, err = capsys.readouterr()
    assert err == ''


def test_predict(capsys):
    forecast = predict(new_interim_df, features[0], scaler, forecaster)
    out, err = capsys.readouterr()
    assert err == ''


forecast = predict(df[-len(new_interim_df):], features[0], scaler, forecaster)
new_forecast = predict(new_interim_df, features[0], scaler, forecaster)
new_forecast_list = [new_forecast, None]


@pytest.mark.parametrize('rolling_window', window_list)
@pytest.mark.parametrize('new_data_value', new_data_list)
@pytest.mark.parametrize('new_forecast_value', new_forecast_list)
@pytest.mark.parametrize('plot_each_unit', plot_each_unit_list)
def test_plot_forecast(capsys, rolling_window, new_data_value,
                       new_forecast_value, plot_each_unit):
    plot_forecast(history_df=df,
                  forecast=forecast,
                  feature=features[0],
                  rolling_window=rolling_window,
                  new_data_df=new_data_value,
                  new_forecast=new_forecast_value,
                  plot_each_unit=plot_each_unit)
    out, err = capsys.readouterr()
    assert err == ''