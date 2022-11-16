import pandas as pd
from streamlit import cache


@cache
def build_time_features(df):
    df['RUNNING_SECONDS'] = (pd.to_timedelta(
        range(len(df)),
        unit='s',
    ).total_seconds()).astype(int)
    df['RUNNING_HOURS'] = (df['RUNNING_SECONDS'] / 3600).astype(float)
    return df
