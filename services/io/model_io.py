from uuid import uuid4
from drashta_types.drashta_types_model import TModelMetadata
from drashta_types.drashta_types_key import MLTaskTypes
from drashta_types.drashta_types_data import TDataSeriesMinimal
from flaml import AutoML
import datetime
import dotenv
import os
import pickle
import json
dotenv.load_dotenv()

def model_meta_data(model_name:str,x_var:list[TDataSeriesMinimal],target:list[TDataSeriesMinimal],task:MLTaskTypes,estimator:str)->TModelMetadata:
    print(f'metadata save targetas : {target}')
    meta_data = TModelMetadata(
        id=str(uuid4()),
        name = model_name,
        columns = x_var,
        target = target,
        date = datetime.datetime.now().strftime("%Y-%m-%d"),
        task = task,
        estimator = estimator,
    )
    return meta_data

def save_model_tensor(model_meta_data:TModelMetadata,model:AutoML):
    
    model_dir = os.getenv("MODEL_DIR")
    print(f'MODEL DIR {model_dir}')
    file_name = f"{model_dir}/{model_meta_data.name}_{model_meta_data.id}.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    return file_name

def load_model_tensor(file_name):
    try:
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def rm_model_tensor(file_name):
    try:
        os.remove(file_name)
    except Exception as e:
        print(f"An error occurred while removing the model: {e}")
