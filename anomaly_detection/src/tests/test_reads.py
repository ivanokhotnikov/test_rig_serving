from components import read_raw_data


def test_read_raw_data(capsys):
    df = read_raw_data()
    out, err = capsys.readouterr()
    assert err == ''