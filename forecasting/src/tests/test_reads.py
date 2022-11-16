from components import (read_latest_unit, read_processed_data, read_raw_data,
                        read_unit_data)


def test_read_processed_data(capsys):
    df = read_processed_data()
    out, err = capsys.readouterr()
    assert err == ''


df = read_processed_data()


def test_read_latest_unit(capsys):
    read_latest_unit(df)
    out, err = capsys.readouterr()
    assert err == ''


def test_read_unit_data(capsys):
    df = read_unit_data('HYD000130-R1_RAW.csv')
    out, err = capsys.readouterr()
    assert err == ''


def test_read_raw_data(capsys):
    df = read_raw_data()
    out, err = capsys.readouterr()
    assert err == ''
