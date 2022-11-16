import pytest
from components import (build_power_features, import_forecast_features,
                        import_forecaster, import_scaler,
                        plot_correlation_matrix, plot_forecast, plot_unit,
                        predict, read_latest_unit, read_processed_data,
                        read_unit_data, remove_step_zero)

df = read_processed_data()
features = import_forecast_features()
new_data_df = read_unit_data('HYD000130-R1_RAW.csv')
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