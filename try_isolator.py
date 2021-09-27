import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

import utils.readers as r
import utils.config as c
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    df = r.load_data(read_all=True, raw=True)
    loaded_steps_df = df.loc[((df['STEP'] >= 12) & (df['STEP'] <= 22)) |
                             ((df['STEP'] >= 24) & (df['STEP'] <= 31))]
    target = 'STEP'
    features = c.FEATURES_NO_TIME_AND_COMMANDS
    features.remove(target)
    loaded_steps_df[target] = loaded_steps_df[target].astype(np.uint8)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    x_train, x_val = train_test_split(
        df[features],
        test_size=0.2,
        shuffle=True,
    )
    isolator = IsolationForest(
        n_estimators=10000,
        warm_start=True,
        n_jobs=-1,
        verbose=100,
    )
    isolator.fit(x_train)
    preds = isolator.predict(x_val)
    x_val.loc[(preds == -1)]
    x_val['ANOMALY'] = pd.Series(preds)
    sns.scatterplot(data=x_val, x='M1 CURRENT', y='M1 TORQUE', hue='ANOMALY')
    sns.scatterplot(data=x_val, x='D1 CURRENT', y='D1 TORQUE', hue='ANOMALY')
    sns.pairplot(x_val, hue='ANOMALY')
    plt.show()


if __name__ == '__main__':
    main()