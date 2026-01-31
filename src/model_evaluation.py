import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path:str) -> pd.DataFrame :
    try :
        df = pd.read_csv(file_path)
        logger.debug('data loaded from the path %s',file_path)
        return df 
    except pd.errors.ParseError as e :
        logger.error('failed to parse the csv file %s',e)
    except Exception as e :
        logger.error('unexpected errors occured %s',e)

def load_model(file_path :str):
    '''load the trained model from the provided path'''
    try:
        with open(file_path,'rb') as file :
            model = pickle.load(file)
        logger.debug('model loaded from provided path %s',file)
        return model 
    except FileNotFoundError as e :
        logger.error('file not found at provided path %s',e)
    except Exception as e:
     logger.error('unexpected error see %s',e)

def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray) -> dict:
    ''' this will evaluate the model and return the performance metrics'''
    try:
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_prob)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug('model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('unexpected error occured while evaluating model %s',e)
        raise 


def save_metrics(metrics : dict,file_path : str) -> None:
    '''save the evaluation metrics to json path'''
    try :
        path = file_path
        os.makedirs(os.path.dirname(path),exist_ok = True)

        with open (path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug('metrics saved to path %s',file)
    except Exception as e:
        logger.error('unusual error %s',e)
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        clf = load_model('/Users/lucky/End-to-End-ML-pipeline-using-DVC/models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        
        #experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
          live.log_metric('accuracy', accuracy_score(y_test, y_test))
          live.log_metric('precision', precision_score(y_test, y_test))
          live.log_metric('recall', recall_score(y_test, y_test))
          live.log_params(params)
        
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
