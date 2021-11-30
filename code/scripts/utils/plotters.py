import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import IMAGES_PATH, FEATURES_NO_TIME, FEATURES_NO_TIME_AND_COMMANDS


class Plotter:
    @staticmethod
    def save_fig(fig_id,
                 tight_layout=True,
                 fig_extension="png",
                 resolution=300):
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    @staticmethod
    def plot_unit_from_summary_file(unit_id='HYD000091-R1'):
        from readers import DataReader

        reader = DataReader()
        units = reader.read_summary_data()
        unit = units[unit_id]
        cols = unit.columns.to_list()
        num_cols = [
            col for col in cols
            if unit[col].dtype in ('float64',
                                   'int64') and col not in ('NOT USED',
                                                            'M4 SETPOINT')
        ]
        fig = make_subplots(rows=len(num_cols),
                            cols=1,
                            subplot_titles=num_cols)
        for i, col in enumerate(num_cols, start=1):
            fig.append_trace(go.Scatter(x=unit['TIME'], y=unit[col]),
                             row=i,
                             col=1)
        fig.update_layout(template='none',
                          height=8000,
                          title=f'{unit_id}',
                          showlegend=False)
        fig.show()

    @staticmethod
    def plot_unit(df, unit=89, save=False, show=True):
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            fig = go.Figure()
            for test in df[df['UNIT'] == unit]['TEST'].unique():
                fig.add_scatter(
                    x=df[(df['UNIT'] == unit)
                         & (df['TEST'] == test)]['TIME'],
                    y=df[(df['UNIT'] == unit)
                         & (df['TEST'] == test)][feature],
                    name=f'{unit}-{test}',
                )
            fig.update_layout(
                template='none',
                title=f'{feature}',
                xaxis_title='TIME',
                yaxis_title=feature,
                showlegend=True,
            )
            if save:
                fig.write_image(
                    os.path.join(IMAGES_PATH,
                                 f'{feature}_unit_{unit}' + '.png'))
            if show: fig.show()

    @staticmethod
    def plot_unit_per_feature(df,
                              unit=89,
                              feature='M4 ANGLE',
                              save=False,
                              show=True):
        for test in df[(df['UNIT'] == unit)]['TEST'].unique():
            Plotter.plot_unit_per_test_feature(df,
                                               unit=unit,
                                               test=test,
                                               feature=feature,
                                               save=save,
                                               show=show)

    @staticmethod
    def plot_unit_per_test_feature(df,
                                   unit=89,
                                   test=1,
                                   feature='M4 ANGLE',
                                   save=False,
                                   show=True):
        fig = go.Figure()
        fig.add_scatter(x=df[(df['UNIT'] == unit)
                             & (df['TEST'] == test)]['TIME'],
                        y=df[(df['UNIT'] == unit)
                             & (df['TEST'] == test)][feature],
                        name=f'{unit}-{test}')
        fig.update_layout(
            template='none',
            title=f'{feature}',
            xaxis_title='TIME',
            yaxis_title=feature,
            showlegend=True,
        )
        if save:
            fig.write_image(
                os.path.join(IMAGES_PATH,
                             f'{feature}_unit_{unit}-{test}' + '.png'))
        if show:
            fig.show()
            return None
        return fig

    @staticmethod
    def plot_unit_per_test_step_feature(df,
                                        unit=89,
                                        test=1,
                                        step=23,
                                        feature='M4 ANGLE',
                                        save=False,
                                        show=True):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df[(df['UNIT'] == unit)
                            & (df['STEP'] == step)
                            & (df['TEST'] == test)]['TIME'],
                       y=df[(df['UNIT'] == unit)
                            & (df['STEP'] == step)
                            & (df['TEST'] == test)][feature],
                       name=f'{unit}-{test}'))
        fig.update_layout(
            template='none',
            title=f'Step {step}, {feature}',
            xaxis_title='TIME',
            yaxis_title=feature,
            showlegend=True,
        )
        if save:
            fig.write_image(
                os.path.join(IMAGES_PATH,
                             f'{feature}_unit_{unit}-{test}-{step}' + '.png'))
        if show:
            fig.show()
            return None
        return fig

    @staticmethod
    def plot_covariance(cls, df, save=False):
        plt.figure(figsize=(10, 10))
        sns.heatmap(df[FEATURES_NO_TIME_AND_COMMANDS].cov(),
                    cmap='RdYlBu_r',
                    linewidths=.5,
                    square=True,
                    cbar=False)
        if save: cls.save_fig('covariance')
        plt.show()

    @classmethod
    def plot_conf_matrix(cls, cm, clf_name, save=False):
        np.fill_diagonal(cm, 0)
        plt.figure(figsize=(18, 18))
        sns.heatmap(cm,
                    annot=True,
                    linewidths=.5,
                    square=True,
                    cbar=False,
                    fmt='d')
        plt.ylabel('True step')
        plt.xlabel('Predicted step')
        if save: cls.save_fig(f'confusion_matrix_{clf_name}')
        plt.show()

    @staticmethod
    def plot_all_per_step_feature(df,
                                  step=18,
                                  feature='M4 ANGLE',
                                  single_plot=True):
        if single_plot:
            fig = go.Figure()
            for unit in df['UNIT'].unique():
                for test in df[(df['UNIT'] == unit)]['TEST'].unique():
                    if not df[(df['TEST'] == test) & (df['STEP'] == step) &
                              (df['UNIT'] == unit)].empty:
                        fig.add_trace(
                            go.Scatter(y=df[(df['UNIT'] == unit)
                                            & (df['STEP'] == step)
                                            & (df['TEST'] == test)][feature],
                                       name=f'{unit}-{test}'))
            fig.update_layout(template='none', title=f'Step {step}, {feature}')
        else:
            titles = [
                f'{unit}-{test}' for unit in df['UNIT'].unique()
                for test in df[df['UNIT'] == unit]['TEST'].unique()
                if not df[(df['TEST'] == test) & (df['STEP'] == step)
                          & (df['UNIT'] == unit)][feature].empty
            ]
            fig = make_subplots(rows=len(titles),
                                cols=1,
                                subplot_titles=titles)
            current_row = 1
            for unit in df['UNIT'].unique():
                for test in df[(df['UNIT'] == unit)]['TEST'].unique():
                    if not df[(df['TEST'] == test) & (df['STEP'] == step) &
                              (df['UNIT'] == unit)].empty:
                        fig.append_trace(
                            go.Scatter(y=df[(df['UNIT'] == unit)
                                            & (df['STEP'] == step)
                                            & (df['TEST'] == test)][feature]),
                            row=current_row,
                            col=1)
                        current_row += 1
            fig.update_layout(template='none',
                              height=12000,
                              title=f'Step {step}, {feature}',
                              showlegend=False)
        fig.show()

    @staticmethod
    def plot_all_per_step(df, step):
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            fig = go.Figure()
            for unit in df['UNIT'].unique():
                for test in df[(df['UNIT'] == unit)
                               & (df['STEP'] == step)]['TEST'].unique():
                    fig.add_trace(
                        go.Scatter(y=df[(df['UNIT'] == unit)
                                        & (df['STEP'] == step)
                                        & (df['TEST'] == test)][feature],
                                   name=f'{unit}-{test}'))
            fig.update_layout(
                template='none',
                title=f'Step {step}, {feature}',
                xaxis_title='Time',
            )
            fig.show()

    @staticmethod
    def plot_all_means_per_step(df, step):
        units = []
        features_means = {
            feature: []
            for feature in FEATURES_NO_TIME_AND_COMMANDS
        }
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            for unit in df['UNIT'].unique():
                for test in df[(df['STEP'] == step)
                               & (df['UNIT'] == unit)]['TEST'].unique():
                    features_means[feature].append(
                        df[(df['STEP'] == step)
                           & (df['UNIT'] == unit) &
                           (df['TEST'] == test)][feature].mean())
                    units.append(unit)
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            fig = go.Figure(data=go.Scatter(
                x=units, y=features_means[feature], mode='markers'))
            fig.update_layout(
                template='none',
                title=f'Step {step}, {feature} means',
                xaxis_title='Unit',
            )
            fig.show()

    @staticmethod
    def plot_kdes_per_step(df, step):
        if 'TIME' in df.columns:
            df.drop('TIME', axis=1, inplace=True)
        fig, axes = plt.subplots(7, 5, figsize=(15, 15))
        axes = axes.flatten()
        for idx, (ax, col) in enumerate(zip(axes, df.columns)):
            sns.kdeplot(
                data=df[df['STEP'] == step],
                x=col,
                fill=True,
                ax=ax,
                warn_singular=False,
            )
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.spines[['top', 'left', 'right']].set_visible(False)
        fig.suptitle(f'STEP {step}')
        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_anomalies_per_unit(df, unit=89):
        if 'ANOMALY' in df.columns:
            for test in df[(df['UNIT'] == unit)]['TEST'].unique():
                for feature in FEATURES_NO_TIME_AND_COMMANDS:
                    Plotter.plot_anomalies_per_unit_test_feature(
                        df, unit=unit, test=test, feature=feature)
        else:
            print(f'No anomalies column in data')

    @staticmethod
    def plot_anomalies_per_unit_feature(df, unit=89, feature='PT4'):
        try:
            for test in df[(df['UNIT'] == unit)]['TEST'].unique():
                Plotter.plot_anomalies_per_unit_test_feature(df,
                                                             unit=unit,
                                                             test=test,
                                                             feature=feature)
        except KeyError:
            print(f'No "ANOMALY" columns found in the dataset.')

    @staticmethod
    def plot_anomalies_per_unit_test_feature(df,
                                             unit=89,
                                             test=1,
                                             feature='M4 ANGLE'):
        if any(df[(df['UNIT'] == unit)
                  & (df['TEST'] == test)]['ANOMALY'] == -1):
            fig = go.Figure()
            fig.add_scatter(
                x=df[(df['UNIT'] == unit)
                     & (df['TEST'] == test)]['TIME'],
                y=df[(df['UNIT'] == unit)
                     & (df['TEST'] == test)][feature],
                mode='lines',
                name='Inlier',
                line={
                    'color': 'steelblue',
                },
            )
            fig.add_scatter(
                x=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
                     (df['ANOMALY'] == -1)]['TIME'],
                y=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
                     (df['ANOMALY'] == -1)][feature],
                mode='markers',
                name='Outlier',
                line={
                    'color': 'indianred',
                },
            )
            fig.update_layout(
                yaxis={'title': feature},
                template='none',
                title=f'Unit {unit}-{test}',
            )
            fig.show()
        else:
            print(f'No anomalies found in {unit}-{test} test.')


if __name__ == '__main__':
    from readers import DataReader

    os.chdir('../../..')
    os.getcwd()
    data = DataReader.load_data()
    Plotter.plot_unit_raw_data(data, unit=91, in_time=False, save=False)
    Plotter.plot_unit_from_summary_file(unit_id='HYD000091-R1')
    Plotter.plot_covariance(data)
    Plotter.plot_all_per_step_feature(data, single_plot=False)
    Plotter.plot_all_per_step(data, step=18)
    Plotter.plot_all_means_per_step(data, step=18)
    Plotter.plot_kdes_per_step(data, step=18)
    Plotter.plot_unit_per_step_feature(data,
                                       unit=91,
                                       step=23,
                                       feature='M4 ANGLE')
    Plotter.plot_unit_per_feature(data, unit=91, feature='M4 ANGLE')
    Plotter.plot_anomalies_per_unit_feature(data, unit=91, feature='M4 ANGLE')