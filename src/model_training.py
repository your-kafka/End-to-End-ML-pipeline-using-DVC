import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict) -> RandomForestClassifier :
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
         raise ValueError('The number of data points in features does not match with labels')
        
        logger.debug('initializing random forest model with parameteres : %s',params)
        clf = RandomForestClassifier(n_estimators = params['n_estimators'],random_state=params['random_state'])
        
        logger.debug('model training started with %d sampels',X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug('model training completed')
        
        return clf 
    except ValueError as e:
        logger.error('value during model training %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured during training',e)
        raise

def save_model(model,file_path:str) -> None :
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok = True)
        
        with open(file_path,'wb') as file : 
            pickle.dump(model,file)
        logger.debug('model saved at file path %s',file_path)
    
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        params = {'n_estimators':25,'random_state':2}
        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf,model_save_path)
            
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
