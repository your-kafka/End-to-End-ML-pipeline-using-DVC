from src.data_ingestion import load_params

params_file = load_params('params.yaml')
params = params_file['model_training']

print(params)
