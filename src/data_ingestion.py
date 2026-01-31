import pandas as pd
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
import logging 
import yaml

#ensure log directory exists 
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
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


def load_data(data_url:str) -> pd.DataFrame :
   '''load data from csv'''
   try : 
      df = pd.read_csv(data_url)
      logger.debug('data load successfully from %s',data_url)
      return df
   except pd.errors.ParseError as e:
      logger.error('Failed to parse the csv file %s',e)
      raise
   except Exception as e :
      logger.error('unexpected errors occured while loading csv %s',e)
      raise
   
def preprocess_data(df:pd.DataFrame) -> pd.DataFrame :
    '''preprocessing data'''
    try : 
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1':'target','v2':'text'},inplace = True)
        logger.debug('data preprocessing completed')
        return df
    except KeyError as e:
       logger.error('Missing clumn in dataframe %s',e)
       raise
    except Exception as e:
       logger.error('unexpected error during preprocessing %s',e)
       raise
    
def save_data(train_data : pd.DataFrame , test_data: pd.DataFrame,file_path:str):
    ''' saving train and test datasets'''
    try : 
       data_path = os.path.join(file_path,'raw')
       os.makedirs(data_path,exist_ok=True)
       train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
       test_data.to_csv(os.path.join(data_path,'test.csv'),index = False)
       logger.debug('train and test data saved to %s',data_path)
    except Exception as e:
       logger.error('error in saving train and test data %s',e)
       raise

def main():
    try :
       params = load_params(params_path='params.yaml')
       test_size = params['data_ingestion']['test_size'] 
       data_path =  'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
       df = load_data(data_path)
       preprocessed_df = preprocess_data(df)
       train_data , test_data = train_test_split(preprocessed_df,test_size=test_size,random_state=2)
       save_data(train_data,test_data,file_path='./data')
    except Exception as e :
       logger.error('Failed to complete data ingestion process %s',e)
       print(f'Error :{e}')

if __name__ == '__main__':
   main()
