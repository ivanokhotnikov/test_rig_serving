import numpy as np
import pytest
from components.get_raw_data_files import get_raw_data_files
from components.get_raw_data_folder_stats import get_raw_data_folder_stats
from components.import_forecast_features import import_forecast_features
from components.read_latest_unit import read_latest_unit
from components.read_processed_data import read_processed_data
from components.read_raw_data import read_raw_data
from components.read_unit_data import read_unit_data


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
    _, files = get_raw_data_folder_stats()
    random_unit_int = np.random.choice(files)
    random_unit_file_name = get_raw_data_files(random_unit_int)[0]
    df = read_unit_data(random_unit_file_name)
    out, err = capsys.readouterr()
    assert err == ''
