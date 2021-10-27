import utils.readers as r
import utils.config as c
import utils.plotters as p


def main():
    df = r.load_data(read_all=True, raw=False, verbose=True)
    df = r.get_performance_check_steps(df)
    for step in df['STEP'].unique():
        p.plot_kdes_per_step(df, step=step)
    p.plot_all_per_step_feature(df, step=27, feature='PT4')
    p.plot_all_per_step_feature(df, step=27, feature='PT4 SETPOINT')
    p.plot_all_per_step_feature(df, step=27, feature='M4 ANGLE')
    p.plot_all_means_per_step(df, step=27)
    p.plot_all_per_step(df, step=27)


if __name__ == '__main__':
    main()