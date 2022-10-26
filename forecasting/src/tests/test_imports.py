import pytest
from components import import_forecast_features, import_metrics, import_model


def test_import_forecast_features(capsys):
    features_str = import_forecast_features()
    out, err = capsys.readouterr()
    assert err == ''


features_list = import_forecast_features()
models = [f + ext for ext in ('.joblib', '.h5') for f in features_list]


@pytest.mark.parametrize('model', models)
def test_import_model(capsys, model):
    import_model(model)
    out, err = capsys.readouterr()
    assert err == ''


@pytest.mark.parametrize('feature', features_list)
def test_import_metrics(capsys, feature):
    import_metrics(feature)
    out, err = capsys.readouterr()
    assert err == ''
