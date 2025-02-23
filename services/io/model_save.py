import safetensors
from flaml import AutoML
import datetime

def model_meta_data(model_name:str,x_var:list[str],target:list[str])->dict:
    return {
        "model_name":model_name,
        "x_variables": x_var,
        "target" : target,
        "created" : datetime.datetime.now().isoformat()
    }

def save_model_tensor(model_name,x_var,target,model:AutoML,file_name:str):
    metadata:dict = model_meta_data(model_name,x_var,target)
    safetensors.serialize_file(
        model,
        file_name,
        metadata
    )
