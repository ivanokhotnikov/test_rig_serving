import logging

import numpy as np

# def create_sequences(values, lookback=None):
#     if lookback is None:
#         logging.error('Look back is not specified')
#     X = []
#     for i in range(lookback, len(values)):
#         X.append(values[i - lookback:i])
#     return np.stack(X)


def create_sequences(values, lookback=None):
    if lookback is None:
        logging.error('Look back is not specified')
    output = []
    for i in range(len(values) - lookback + 1):
        output.append(values[i:(i + lookback)])
    return np.stack(output)
