from pydantic import BaseModel
from typing import List


class MLModelTable(BaseModel):
    model_name:str
    model_file_path:str
    model_data_columns:List[str]
    model_target:str
    model_task_type:str
    model_eval_metric_info:dict
    model_estimator_type:str
    model_optimization_metric:str