import logging

import numpy as np
from streamlit import cache


@cache
def create_sequences(values, lookback=None):
    """
    The create_sequences function creates sequences from the input data. The function takes two arguments: values and lookback. Values is a list of floats, lookback is an integer. The output of the function is a numpy array with shape (number_of_sequences, lookback, 1).
    
    Args:
        values: Pass the data to be converted into sequences
        lookback=None: Specify the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to none
    
    Returns:
        A numpy array
    """
    if lookback is None:
        logging.error('Look back is not specified')
    output = []
    for i in range(len(values) - lookback + 1):
        output.append(values[i:(i + lookback)])
    return np.stack(output)
