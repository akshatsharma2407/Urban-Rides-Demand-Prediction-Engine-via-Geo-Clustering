import pandas as pd
import numpy as np
import logging
import os
import yaml
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(level='DEBUG')

consolehandler = logging.StreamHandler()
consolehandler.setLevel(level='DEBUG')

filehandler = logging.FileHandler('reports/error.log')
filehandler.setLevel(level='DEBUG')

formatter = formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

filehandler.setFormatter(formatter)
consolehandler.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(consolehandler)

def save_model(model, path):
    joblib.dump(model, path)

def load_params(param_path = 'params.yaml'):
    return yaml.safe_load(open(param_path, 'r'))['XGB']

def load_train_data(data_path = 'data/processed/train.csv'):
    df = pd.read_csv(data_path, parse_dates=['tpep_pickup_datetime'])
    logger.info('train data loaded')

    df.set_index('tpep_pickup_datetime', inplace=True)

    xtrain = df.drop(columns=['total_pickups'])
    ytrain = df['total_pickups'].copy()
    return xtrain, ytrain

def create_encoder(xtrain):
    encoder = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(drop='first', sparse_output=False), ['region', 'day_of_week', 'hour_of_day', 'is_weekend'])
        ],
        remainder='passthrough'
    )
    encoder.set_output(transform='pandas')

    encoder.fit(xtrain)
    logger.info("Encoder trained")
    return encoder

def train_model(xtrain, ytrain, encoder, params):
    ytrain_log = np.log1p(ytrain)

    xgbr = XGBRegressor(**params)

    pipe = Pipeline(
        steps=[
            ('encoder', encoder),
            ('model', xgbr)
        ]
    )
    
    pipe.fit(xtrain, ytrain_log)
    logger.info("Model trained successfully")
    return pipe

if __name__ == "__main__":
    xtrain, ytrain = load_train_data()
    params = load_params()
    encoder = create_encoder(xtrain)

    pipe = train_model(xtrain, ytrain, encoder, params)

    save_model(pipe, 'models/model.joblib')
    logger.info('model saved !')