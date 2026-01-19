import dask.dataframe as dd
import logging
import os
import yaml

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

def load_params(params_path: str):
    try:
          params = yaml.safe_load(open(params_path, 'r'))['data_ingestion']
          return params
    except:
         pass

def read_dask_dfs(df_path,  parse_dates: list = ['tpep_pickup_datetime'], columns: list=['trip_distance', 
                                'tpep_pickup_datetime', 
                                'pickup_longitude',
                                'pickup_latitude',
                                'dropoff_longitude', 
                                'dropoff_latitude', 
                                'fare_amount']):
    
    dfs = []
    for perticular_df in df_path:
        df = dd.read_csv(perticular_df, parse_dates=parse_dates, usecols=columns)
        dfs.append(df)
    logger.info("Dask DataFrames are read successfully!!")

    df_final = dd.concat(dfs, axis=0)
    logger.info('All datasets are concatinated successfully')
    return df_final

def outlier_removal(df, params):
        df = (
            df
            .loc[
                (
                    # removing coordinate that are not inside/on bounding box 
                    df['pickup_latitude'].between(params['min_latitude'], params['max_latitude'], inclusive='both') & 
                    df['pickup_longitude'].between(params['min_longitude'], params['max_longitude'], inclusive='both') &
                    df['dropoff_latitude'].between(params['min_latitude'], params['max_latitude'], inclusive='both') &
                    df['dropoff_longitude'].between(params['min_longitude'], params['max_longitude'], inclusive='both') &

                    # removing outliers present in fare and trip distance
                    df['fare_amount'].between(params['min_fare'], params['max_fare'], inclusive='both') &
                    df['trip_distance'].between(params['min_distance'], params['max_distance'], inclusive='both')
                ),
                :
            ]
        )

        logger.info("outliers are removed successfully")

        cols_to_drop = ['trip_distance', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']
        df = df.drop(columns = cols_to_drop)

        logger.info("Columns dropped successfully")
        df = df.compute()

        logger.info('dask dataframe is computed successfully!!')
        return df

if __name__ == "__main__":
    params = load_params('params.yaml')
    df_path = ["data/raw/yellow_tripdata_2016-01.csv",
                "data/raw/yellow_tripdata_2016-02.csv",
                "data/raw/yellow_tripdata_2016-03.csv"]
    
    df_final = read_dask_dfs(df_path)

    df_final = outlier_removal(df_final, params)
    logger.info("Dast pipeline is executed successfully")

    df_without_outliers_path = 'data/interim/df_without_outliers.csv'
    df_final.to_csv(df_without_outliers_path, index=False)
    logger.info("Dataframe after Outlier removal is saved successfully!!")
     