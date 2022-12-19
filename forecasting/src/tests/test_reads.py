from components import (read_latest_unit, read_processed_data, read_raw_data,
                        read_unit_data, import_forecast_features)
import pytest


def test_read_raw_data(capsys):
    df = read_raw_data()
    out, err = capsys.readouterr()
    assert err == ''


features = import_forecast_features()


@pytest.mark.parametrize('feature', features)
def test_read_processed_data(capsys, feature):
    df = read_processed_data(features=[feature])
    out, err = capsys.readouterr()
    assert err == ''


df = read_processed_data(features=features + ['UNIT', 'TEST', 'TIME'])


def test_read_latest_unit(capsys):
    latest = read_latest_unit(df)
    out, err = capsys.readouterr()
    assert err == ''


def test_read_unit_data(capsys):
    df = read_unit_data('HYD000130-R1_RAW.csv')
    out, err = capsys.readouterr()
    assert err == ''
