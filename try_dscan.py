import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import seaborn as sns

import utils.readers as r
import utils.plotters as p
import utils.config as c


def main():
    df = r.load_data(read_all=True, raw=True)
    target = 'STEP'
    df[target] = df[target].astype(np.uint8)
    features = c.FEATURES_NO_TIME_AND_COMMANDS
    x_train, x_val = train_test_split(
        df[features],
        test_size=0.2,
        shuffle=True,
    )
    clustering = DBSCAN().fit(X=x_val)
    uniue_labels = set(clustering.labels_)


if __name__ == '__main__':
    main()
