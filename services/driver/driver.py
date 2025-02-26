from turtle import mode
from ..automl.model_generation import generate_model
from ..io.model_save import save_model_tensor, model_meta_data
from ..bridge.data_retrieval import get_table_dataframe
from drashta_types.drashta_types_key import MLTaskTypes
from flaml.automl import AutoML
from pydantic import BaseModel

class MLModelQuery(BaseModel):
    table: str
    x: list[str]
    y: str
    model_name: str
    size: int = 1000
    task: MLTaskTypes = MLTaskTypes.CLASSIFICATION

def MLModelQueryHandle(
    table:str,
    x:list[str],
    y:str,
    model_name:str,
    size:int = 1000,
    task:MLTaskTypes = MLTaskTypes.CLASSIFICATION,
    ):
    cols = x + [y]
    data = get_table_dataframe(table,cols,size)
    model:AutoML = generate_model(data,x,y,task)
    meta_data = model_meta_data(
        model_name,
        x,
        y,
        task,
        model.best_estimator
    )
    model = save_model_tensor(meta_data,model)
    return model
