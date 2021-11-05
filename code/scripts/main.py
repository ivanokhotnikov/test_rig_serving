from utils.readers import load_data, get_performance_check_steps
from utils.plotters import plot_kdes_per_step, plot_all_per_step_feature, plot_all_per_step, plot_all_means_per_step


def main():
    df = load_data(read_all=True, raw=False, verbose=True)
    df = get_performance_check_steps(df)


if __name__ == '__main__':
    main()