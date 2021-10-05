import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import readers as r
from . import config as c


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(c.IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_all(df):
    fig = make_subplots(rows=len(df[c.FEATURES_NO_TIME]),
                        cols=1,
                        subplot_titles=df[c.FEATURES_NO_TIME])
    for i, col in enumerate(df[c.FEATURES_NO_TIME], start=1):
        fig.append_trace(go.Scatter(y=df[col]), row=i, col=1)
    fig.update_layout(template='simple_white', height=12000, showlegend=False)
    fig.show()


def plot_unit_from_summary_file(unit_id='HYD000091-R1'):
    units = r.read_summary_file()
    unit = units[unit_id]

    cols = unit.columns.to_list()
    num_cols = [
        col for col in cols
        if unit[col].dtype in ('float64',
                               'int64') and col not in ('NOT USED',
                                                        'M4 SETPOINT')
    ]
    fig = make_subplots(rows=len(num_cols), cols=1, subplot_titles=num_cols)
    for i, col in enumerate(num_cols, start=1):
        fig.append_trace(go.Scatter(x=unit['TIME'], y=unit[col]), row=i, col=1)
    fig.update_layout(template='simple_white',
                      height=8000,
                      title=f'{unit_id}',
                      showlegend=False)
    fig.show()


def plot_unit_raw_data(unit_df):
    fig = make_subplots(rows=len(c.FEATURES_NO_TIME),
                        cols=1,
                        subplot_titles=c.FEATURES_NO_TIME)
    for i, feature in enumerate(c.FEATURES_NO_TIME, start=1):
        fig.append_trace(go.Scatter(x=unit_df['TIME'], y=unit_df[feature]),
                         row=i,
                         col=1)
    fig.update_layout(template='simple_white', height=8000, showlegend=False)
    fig.show()


def plot_covariance(df):
    plt.figure(figsize=(10, 10))
    sns.heatmap(df[c.FEATURES_NO_TIME_AND_COMMANDS].cov(),
                cmap='RdYlBu_r',
                linewidths=.5,
                square=True,
                cbar=False)
    save_fig('covariance')
    plt.show()


def plot_conf_matrix(cm, clf_name):
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
    save_fig(f'confusion_matrix_{clf_name}')
    plt.show()


def plot_all_per_step_feature(df, step=18, feature='PT4', single_plot=True):
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
        fig.update_layout(template='simple_white',
                          title=f'Step {step}, {feature}')
    else:
        titles = [
            f'{unit}-{test}' for unit in df['UNIT'].unique()
            for test in df[df['UNIT'] == unit]['TEST'].unique()
            if not df[(df['TEST'] == test) & (df['STEP'] == step)
                      & (df['UNIT'] == unit)][feature].empty
        ]
        fig = make_subplots(rows=len(titles), cols=1, subplot_titles=titles)
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
        fig.update_layout(template='simple_white',
                          height=12000,
                          title=f'Step {step}, {feature}',
                          showlegend=False)
    fig.show()


def plot_all_per_step(df, step):
    for feature in c.FEATURES_NO_TIME_AND_COMMANDS:
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
            template='simple_white',
            title=f'Step {step}, {feature}',
            xaxis_title='Time',
        )
        fig.show()


def plot_all_means_per_step(df, step):
    units = []
    features_means = {
        feature: []
        for feature in c.FEATURES_NO_TIME_AND_COMMANDS
    }
    for feature in c.FEATURES_NO_TIME_AND_COMMANDS:
        for unit in df['UNIT'].unique():
            for test in df[(df['STEP'] == step)
                           & (df['UNIT'] == unit)]['TEST'].unique():
                features_means[feature].append(
                    df[(df['STEP'] == step)
                       & (df['UNIT'] == unit) &
                       (df['TEST'] == test)][feature].mean())
                units.append(unit)
    for feature in c.FEATURES_NO_TIME_AND_COMMANDS:
        fig = go.Figure(data=go.Scatter(
            x=units, y=features_means[feature], mode='markers'))
        fig.update_layout(
            template='simple_white',
            title=f'Step {step}, {feature} means',
            xaxis_title='Unit',
        )
        fig.show()


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