import joblib
import pandas as pd
import yaml
import logging
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import os
import json

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(level='DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel(level='DEBUG')

file_handler = logging.FileHandler('reports/error.log')
file_handler.setLevel(level='DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def read_cluster_input(data_path = 'data/interim/df_without_outliers.csv', chunksize=100000, usecols=["pickup_latitude","pickup_longitude"]):
    df_reader = pd.read_csv(data_path, chunksize=chunksize, usecols=usecols)
    logger.info("Data read successfully")
    return df_reader

def scaling_coordinates_and_saving_artifacts():
    df_reader = read_cluster_input()
    scaler = StandardScaler()

    for chunk in df_reader:
        scaler.partial_fit(chunk)
    
    scaler_save_path = 'models/scaler.joblib'
    save_model(scaler, scaler_save_path)
    logger.info("scaler saved successfully")
    return scaler

def train_KMeans(mini_batch_params, scaler):
    mini_batch = MiniBatchKMeans(**mini_batch_params)
    df_reader = read_cluster_input()

    for chunk in df_reader:
        scaled_chunk = scaler.transform(chunk)
        mini_batch.partial_fit(scaled_chunk)

    return mini_batch

def save_model(model, save_path):
    joblib.dump(model, save_path)
    logger.info('Kmeans model saved successfully')

def read_params(params_path = 'params.yaml'):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)['build_features']
    return params

def assign_cluster(scaler, mini_batch, data_path = 'data/interim/df_without_outliers.csv'):
    df_final = pd.read_csv(data_path, parse_dates=['tpep_pickup_datetime'])
    logger.info('Data Read for cluster predictions')
    location_subset = df_final.loc[:, ['pickup_longitude', 'pickup_latitude']]
    scaled_location_subset = scaler.transform(location_subset)
    cluster_predictions = mini_batch.predict(scaled_location_subset)
    df_final['region'] = cluster_predictions
    logger.info('cluster predictions are added to data')
    return df_final

def extract_features(df_final):
    df_final = df_final.drop(columns=['pickup_longitude', 'pickup_latitude'])
    logger.info("latitude and longitude columns are dropped")

    df_final.set_index('tpep_pickup_datetime', inplace=True)

    region_grp = df_final.groupby('region')

    resampled_data = (
        region_grp['region']
        .resample('15min')
        .count()
    )

    logger.info("Data converted to 15 min intervals successfully")

    resampled_data.name = 'total_pickups'

    resampled_data = resampled_data.reset_index(level=0)

    epsilon = 10
    resampled_data.replace({'total_pickups' : {0 : epsilon}}, inplace=True)

    resampled_data['avg_pickups'] = resampled_data.groupby('region')['total_pickups'].ewm(alpha=0.4).mean().round().values

    resampled_data['avg_pickups'] = resampled_data.groupby('region')['avg_pickups'].shift(1)

    resampled_data.dropna(inplace=True)
    logger.info("average pickups calculated successfully using EWMA")

    resampled_data.reset_index(inplace=True)

    resampled_data['day_of_week'] = resampled_data['tpep_pickup_datetime'].dt.day_of_week

    resampled_data['month'] = resampled_data['tpep_pickup_datetime'].dt.month

    resampled_data['hour_of_day'] = resampled_data['tpep_pickup_datetime'].dt.hour

    resampled_data['is_weekend'] = resampled_data['day_of_week'].isin([5,6]).values.astype(int)

    resampled_data['pickups_same_time_yesterday'] = resampled_data.groupby(['region'])['total_pickups'].shift(96)

    resampled_data['last_4_days_std'] = resampled_data.groupby(['region'])['total_pickups'].rolling(window=4).std().values

    resampled_data['last_4_days_std'] = resampled_data['last_4_days_std'].shift(1)

    resampled_data['lag_1'] = resampled_data.groupby(['region'])['total_pickups'].shift(1)
    resampled_data['lag_2'] = resampled_data.groupby(['region'])['total_pickups'].shift(2)
    resampled_data['lag_3'] = resampled_data.groupby(['region'])['total_pickups'].shift(3)
    resampled_data['lag_4'] = resampled_data.groupby(['region'])['total_pickups'].shift(4)

    resampled_data.dropna(inplace=True)

    resampled_data.set_index('tpep_pickup_datetime', inplace=True)

    logger.info("Lag/temporal Features generated successfully")
    return resampled_data

def train_test_split_and_save(resampled_data):
    train = resampled_data.loc[resampled_data['month'].isin([1,2]), :]
    test = resampled_data.loc[resampled_data['month'].isin([3]), :]
    logger.info("train and test set created successfully")

    train.to_csv('data/processed/train.csv', index=True)
    test.to_csv('data/processed/test.csv', index=True)
    logger.info('train and test set are saved !') 

def save_coordinates(mini_batch, scaler):
    centroids = scaler.inverse_transform(mini_batch.cluster_centers_)
    dic = {}
    for i, centre in enumerate(centroids):
        dic[i] = centre.tolist()
    with open('coordinates.json', 'w') as f:
        json.dump(dic, f)
    logger.info('centroids coordinates saved successfully!!')

if __name__ == "__main__":

    scaler = scaling_coordinates_and_saving_artifacts()
    params = read_params()
    mini_batch = train_KMeans(params, scaler)
    save_coordinates(mini_batch, scaler)
    save_model(mini_batch, 'models/mbkmeans.joblib')
    final_df = assign_cluster(scaler, mini_batch)
    resampled_data = extract_features(final_df)
    train_test_split_and_save(resampled_data)