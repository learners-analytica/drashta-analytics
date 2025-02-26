import safetensors
import uuid
from drashta_types.drashta_types_model import TModelMetadata
from drashta_types.drashta_types_key import MLTaskTypes
from flaml import AutoML
import datetime
import dotenv
import os

dotenv.load_dotenv()

def model_meta_data(model_name:str,x_var:list[str],target:list[str],task:MLTaskTypes,estimator:str)->TModelMetadata:
    meta_data = TModelMetadata(
        id=uuid.uuid4(),
        model_name = model_name,
        data = x_var,
        target = target,
        date = datetime.datetime.now().strftime("%Y-%m-%d"),
        task = task,
        estimator = estimator,
    )
    return meta_data

def save_model_tensor(model_meta_data:TModelMetadata,model:AutoML):
    model_dir = os.environ("MODEL_DIR")
    file_name = f"{model_dir}/{model_meta_data.model_name}_{model_meta_data.id}.safetensors"
    safetensors.serialize_file(
        model,
        file_name,
        model_meta_data
    )
    return file_name

def load_model_tensor(file_name):
    return safetensors.deserialize(file_name)
