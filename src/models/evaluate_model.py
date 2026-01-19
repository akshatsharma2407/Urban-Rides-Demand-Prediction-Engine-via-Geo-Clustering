import mlflow
import dagshub
import numpy as np
import json
import pandas as pd
import joblib
import logging
import os

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/Urban-Rides-Demand-Prediction-Engine-via-Geo-Clustering.mlflow')
dagshub.init(repo_owner='akshatsharma2407', repo_name='Urban-Rides-Demand-Prediction-Engine-via-Geo-Clustering', mlflow=True)

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(level='DEBUG')

filehandler = logging.FileHandler('reports/error.log')
filehandler.setLevel(level='DEBUG')

consolehandler = logging.StreamHandler()
consolehandler.setLevel(level='DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

filehandler.setFormatter(formatter)
consolehandler.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(consolehandler)

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def save_run_information(run_id, artifact_path, model_uri, path):
    run_information = {
        "run_id" : run_id,
        "artifact_path" : artifact_path,
        "model_uri" : model_uri,
        "path" : path
    }
    with open(path, "w") as f:
        json.dump(run_information, f, indent=4)

def SMAPE(y_true, y_pred):
    """
    custom SMAPE loss (Symmetrical Mean Absolute Percentage Error)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred))/2
    diff = np.abs(y_true - y_pred) / denominator
    return np.mean(diff)

def load_test_data(data_path = 'data/processed/train.csv'):
    test_data = pd.read_csv(data_path, parse_dates=['tpep_pickup_datetime'])
    test_data.set_index("tpep_pickup_datetime", inplace=True)
    logger.info("Data read successfully")

    xtest = test_data.drop(columns=['total_pickups'])
    ytest = test_data['total_pickups'].copy()
    return xtest, ytest

def load_artifacts( model_path = 'models/model.joblib'):

    model_path = 'models/model.joblib'
    model = load_model(model_path)
    logger.info("Data transformed successfully")
    return model

def eval_model(model, xtest, ytest):

    logger.info("Data transformed successfully")

    ypred_log = model.predict(xtest)
    ypred = np.expm1(ypred_log)

    loss = SMAPE(ytest, ypred)
    logger.info(f'loss : {loss}')
    return loss

if __name__ == "__main__":
    xtest,ytest = load_test_data()
    model = load_artifacts()
    loss = eval_model(model, xtest, ytest)

    with mlflow.start_run(run_name='model'):
        mlflow.log_params(model.get_params())
        mlflow.log_metric('SMAPE', loss)

        model_signature = mlflow.models.infer_signature(xtest, ytest)

        logged_model = mlflow.sklearn.log_model(model, "demand_prediction",
                                                signature=model_signature,
                                                pip_requirements='requirements.txt')
    
    run_id = logged_model.run_id
    artifact_path = logged_model.artifact_path
    model_uri = logged_model.model_uri
    logger.info("mlflow logging completed")

    json_file_save_path = 'reports/run_info.json'
    save_run_information(run_id, artifact_path, model_uri, json_file_save_path)
    logger.info('Run information saved successfully')