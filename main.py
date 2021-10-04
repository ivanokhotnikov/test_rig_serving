import numpy as np

from scipy import stats

import utils.readers as r
import utils.config as c
import utils.plotters as p


def remove_outliers(df, z_score):
    return df[(np.abs(stats.zscore(df[c.FEATURES_NO_TIME])) < z_score).all(
        axis=1)]


def remove_step_zero(df):
    return df.drop(df[df['STEP'] == 0].index, axis=0)


def get_warm_up_steps(df):
    return df[(df['STEP'] >= 1) & (df['STEP'] <= 11)]


def get_break_in_steps(df):
    return df[(df['STEP'] >= 12) & (df['STEP'] <= 22)]


def get_performance_check_steps(df):
    return df[(df['STEP'] >= 23) & (df['STEP'] <= 33)]


def main():
    df = r.load_data(read_all=True, raw=False, verbose=True)
    df = remove_outliers(df, z_score=5)
    df = get_performance_check_steps(df)
    p.plot_per_step(df, step=29)
    p.plot_means_per_step(df, step=29)


if __name__ == '__main__':
    main()
