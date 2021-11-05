from .config import MODELS_PATH


def save_data(df):
    """
    Save dataframe to csv file
    """
    df.to_csv(MODELS_PATH + 'data.csv', index=False)
