import pandas as pd


def build_time_features(df):
    df['TIME'] = pd.to_datetime(
        range(len(df)),
        unit='s',
        origin=f'{df[" DATE"].min()} 00:00:00',
    )
    df['DURATION'] = pd.to_timedelta(
        range(len(df)),
        unit='s',
    )
    df['RUNNING_SECONDS'] = (pd.to_timedelta(
        range(len(df)),
        unit='s',
    ).total_seconds()).astype(int)
    df['RUNNING_HOURS'] = (df['TOTAL_SECONDS'] / 3600).astype(float)
    return df
