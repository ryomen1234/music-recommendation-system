import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yaml


# Setup logging
log_dir = Path('logs')
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(log_dir / 'model_training.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_param(param_path: str) -> dict:
    '''this functin load parametrs from param.yaml file'''
    try:
        with open(param_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error("File Not Fund: ", load_param)
        raise
    except Exception as e:
        logger.error("Erro occured: ",e)
        raise


def model_training(x_train: np.ndarray, param: dict) -> KMeans:
    """Train a KMeans clustering model with given training data and parameters."""
    try:
        logger.debug("Starting preprocessing with StandardScaler.")
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)

        logger.debug("Initializing KMeans with parameters: %s", param)
        model = KMeans(**param)
        model.fit(x_train_scaled)

        logger.info("KMeans training completed.")
        return model

    except Exception as e:
        logger.exception("Model training failed.")
        raise


def save_model(model: KMeans, path: Path) -> None:
    """Save the trained model as a pickle file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved at %s", path)

    except Exception as e:
        logger.exception("Failed to save model.")
        raise


def main():
    try:
        params = load_param('params.yaml')
        test_size = params['model']['test_size']
        # Parameters
        params = params['model']['parameter']

        data_path = Path('data/clean_data/clean_data1.csv')
        df = pd.read_csv(data_path)
        logger.info("Data loaded from %s", data_path)

        x_train, x_test = train_test_split(df, test_size=test_size, random_state=42)
        logger.debug("Data split into training and test sets.")

        save_dir = Path('data/clean_data')
        save_dir.mkdir(parents=True, exist_ok=True)
        x_train.to_csv(save_dir / 'x_train.csv', index=False)
        x_test.to_csv(save_dir / 'x_test.csv', index=False)
        logger.info("Train-test data saved.")

        model = model_training(x_train, params)

        model_path = Path('models/model.pkl')
        save_model(model, model_path)

    except Exception as e:
        logger.exception("An error occurred in main.")
        raise


if __name__ == "__main__":
    main()
