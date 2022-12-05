import json

import pandas as pd

from components.constants import FEATURES_BUCKET


def is_schema_valid(uploaded_file):
    """
    The is_schema_valid function checks whether the uploaded CSV file has the correct schema. It returns True if it does, and False otherwise.
    
    Args:
        uploaded_file: Pass the file uploaded by the user
    
    Returns:
        A boolean value
    """
    df = pd.read_csv(uploaded_file, index_col=False, header=0)
    interim_features_blob = FEATURES_BUCKET.get_blob('interim_features.json')
    interim_features_list = list(
        json.loads(interim_features_blob.download_as_string()))
    return all([f in df.columns for f in interim_features_list[:-2]])