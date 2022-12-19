import logging

import numpy as np


def create_sequences(values, lookback=None, inference=False):
    """
    The create_sequences function creates a set of input sequences from the input data. The function takes as input: values: the original array of floating point data, lookback: how many timesteps back should our input data go. Returns two arrays: X - an array of sequences that will be used as inputs to the model, where each sequence is a NumPy array with shape (lookback, 1), Y - an array of expected outputs for each sequence.
    
    Args:
        values: Pass the data that will be used to create the sequences
        lookback=None: Specify the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to none
        inference=False: Specify that the function is used for training
    
    Returns:
        The x and y values
    """
    if lookback is None:
        logging.error('Look back is not specified')
    X, Y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        Y.append(values[i])
    if inference:
        return np.stack(X)
    return np.stack(X), np.stack(Y)
