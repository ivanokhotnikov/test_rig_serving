import pytest
from components.import_forecast_features import import_forecast_features
from components.import_forecaster import import_forecaster
from components.import_metrics import import_metrics
from components.import_scaler import import_scaler


def test_import_forecast_features(capsys):
    features_str = import_forecast_features()
    out, err = capsys.readouterr()
    assert err == ''


forecast_features = import_forecast_features()


@pytest.mark.parametrize('feature', forecast_features)
def test_import_scaler(capsys, feature):
    scaler = import_scaler(feature)
    out, err = capsys.readouterr()
    assert err == ''


@pytest.mark.parametrize('feature', forecast_features)
def test_import_forecaster(capsys, feature):
    forecaster = import_forecaster(feature)
    out, err = capsys.readouterr()
    assert err == ''


@pytest.mark.parametrize('feature', forecast_features)
def test_import_metrics(capsys, feature):
    import_metrics(feature)
    out, err = capsys.readouterr()
    assert err == ''
