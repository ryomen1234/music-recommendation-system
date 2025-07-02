import pandas as pd
import os
import logging 

#ensuring the logs directory exit
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

#console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#file handler
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#set formatter
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

#attach formatter to console and file handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#attach to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path: str) -> pd.DataFrame:
    '''load data from path given'''
    try:
        df = pd.read_csv(path)
        logger.debug("data loading is successful.")
        return df
    except FileNotFoundError as e:
        logger.error('file not found error: %s',e)
        raise
    except Exception as e:
        logger.error('unexcepted error occured during loading data: %s', e)
        raise
    

def save_data(df: pd.DataFrame, path: str) -> None:
    '''save data to data/raw folder'''
    try:
        raw_data_path = os.path.join(path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path, 'raw_data.csv'), index=False)
        logger.debug('data is saved successfulyy.')
    except Exception as e:
        logger.error('Some unexepected error occured: %s', e)
        raise
    

    


def main():
    '''data path is given to load_data which load csv file and save_data store it in ./data/raw file'''
    try:
        path = "E:\project\Music Recommendation System\Data\msd_features.csv"
        logger.debug("path is loaded successfully: %s",path)
        df = load_data(path)
        save_data(df, './data')
    except Exception as e:
        logger.error('Failed to complete data injestion process: %s', e)
        raise


if __name__ == "__main__":
    main()



