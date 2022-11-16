import logging

import numpy as np
from streamlit import cache


@cache
def create_sequences(values, lookback=None, inference=False):
    if lookback is None:
        logging.error('Look back is not specified')
    X, Y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        Y.append(values[i])
    if inference:
        return np.stack(X)
    return np.stack(X), np.stack(Y)
