import pytest
from components import (plot_correlation_matrix, plot_forecast, plot_unit,
                        read_latest_unit, import_forecast_features,
                        read_unit_data, remove_step_zero, build_power_features,
                        read_processed_data, import_model, predict)

df = read_processed_data()
features = import_forecast_features()
new_data_df = read_unit_data('HYD000130-R1_RAW.csv')
new_data_df_wo_zero = remove_step_zero(new_data_df)
new_interim_df = build_power_features(new_data_df_wo_zero)
latest_unit_df = read_latest_unit(df)
new_list = [new_interim_df, None]
window_list = [3200, None]
plot_each_unit_list = [True, False]
forecaster = import_model(f'{features[0]}.h5')
scaler = import_model(f'{features[0]}.joblib')


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


forecast = predict(new_interim_df, features[0], scaler, forecaster)


@pytest.mark.parametrize('rolling_window', window_list)
@pytest.mark.parametrize('new', new_list)
@pytest.mark.parametrize('plot_each_unit', plot_each_unit_list)
def test_plot_forecast(capsys, rolling_window, new, plot_each_unit):
    plot_forecast(df,
                  forecast,
                  features[0],
                  rolling_window=rolling_window,
                  new=new_interim_df,
                  plot_each_unit=plot_each_unit)
    pass