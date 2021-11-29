import datetime
from config import MODELS_PATH
from joblib import dump, load


def save_data(df):
    """
    Save dataframe to csv file
    """
    df.to_csv(MODELS_PATH + 'data.csv', index=False)


def save_model(model):
    """
    Save model to pickle file
    """
    dump(model, MODELS_PATH + 'model.joblib')


def seve_predictions(predictions):
    """
    Save predictions to csv file
    """
    predictions.to_csv(MODELS_PATH +
                       f'predictions_{datetime.datetime.now():%d%m_%I%M}.csv',
                       index=False)
