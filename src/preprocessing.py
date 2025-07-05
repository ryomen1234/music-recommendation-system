import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


import os
import logging as logging


#create logs dir 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#define logger object
logger = logging.getLogger('preprocessing')
logger.setLevel('DEBUG')

#define console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#define file handler
log_file_path = os.path.join(log_dir, 'pre-processing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#defien formatter
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

#attach formatter to console and file handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#attach console and file handler to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def encode(X: dict) -> dict:
        '''this function encode based on numbers of categories'''
        for key, value in X.items():
            if value <= 5:
                X[key] = 0
            elif 5 < value <= 10:
                X[key] = 1
            elif 10 < value <= 15:
                X[key] = 2
            else:
                X[key] = 3
        return X

def encode2(X: str, dictionary: dict) -> int:
  if X in dictionary:
    return dictionary[X]

def pre_processing(df: pd.DataFrame ) -> pd.DataFrame:
    '''this function uses approprite pre-processing techqnic on dataset.'''
    try:
        logger.info("pre-processing started.")
        df.drop(columns=['track_id','title','song_id'], inplace=True)
        logger.info("unwanted columns removed.")

        dict = df['release'].value_counts().to_dict()
        dict2 = df['artist_name'].value_counts().to_dict()
        logger.info("dictionary is created.")

        dict_mod = encode(dict)
        dict_mod2 = encode(dict2)
        logger.debug('dict are encoded.')
        
        df['release'] = df['release'].apply(lambda x: encode2(x, dict_mod))
        df['artist_name'] = df['artist_name'].apply(lambda x: encode2(x, dict_mod2))
        logger.debug('reales and artist name encoded')

        pt = PowerTransformer()
        df[['duration', 'loudness']] = pt.fit_transform(df[['duration', 'loudness']])
        logger.debug('transformation applied on loudness and duration.')

        return df
    except Exception as e:
        logger.error(f"error occured: {e}")
        raise


def main():
    try:
        data_path = './data/raw/raw_data.csv'
        df = pd.read_csv(data_path)
        logger.debug("data loading is done.")

        df = pre_processing(df)
        logger.debug('pre-processing is done.')

        clean_data_path = os.path.join('./data', 'clean_data')
        os.makedirs(clean_data_path, exist_ok=True)
        df.to_csv(os.path.join(clean_data_path, 'clean_data1.csv'), index=False)
        logger.debug(f'clean data is stored in {clean_data_path}')
    except Exception as e:
        logger.error('error occured: {e}')
        raise

if __name__ == "__main__":
    main()


    

