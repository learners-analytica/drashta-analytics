import pandas
import numpy
from services.automl.model_generation import generate_model, predict_model
from services.io.model_io import save_model_tensor, model_meta_data, load_model_tensor
from services.bridge.data_retrieval import get_table_dataframe
from services.io.model_database_CURD import fetch_model_data, Model_DB_Fields, add_new_model
from drashta_types.drashta_types_key import MLTaskTypes
from drashta_types.drashta_types_data import TDataArray
from flaml.automl import AutoML
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

class TMLModelQuery(BaseModel):
    table: str
    x: list[str]
    y: str
    model_name: str
    size: int = 1000
    task: MLTaskTypes = MLTaskTypes.CLASSIFICATION

class TModelPredictRequest(BaseModel):
    x: TDataArray
    model_id: str

async def MLModelQueryHandle(
    table:str,
    x_columns:list[str],
    y:str,
    model_name:str,
    size:int = 1000,
    task:MLTaskTypes = MLTaskTypes.CLASSIFICATION,
    ):
    cols = x_columns + [y]
    data = await get_table_dataframe(table,cols,size)
    model:AutoML = generate_model(data,x_columns,y,task.value)
    meta_data = model_meta_data(
        model_name,
        x_columns,
        y,
        task,
        model.best_estimator
    )
    model_filename = save_model_tensor(meta_data,model)
    add_new_model(meta_data,model_filename)
    return meta_data

async def MLPredictHandle(
    x:TDataArray,
    model_id:str
    ):
    model_data:Model_DB_Fields = fetch_model_data(model_id)
    print(model_data.file_path)
    model_tensor = load_model_tensor(model_data.file_path)
    data = pandas.DataFrame(x)
    preds = predict_model(model_tensor,data)
    if isinstance(preds, numpy.ndarray):
        preds = preds.tolist()
    elif isinstance(preds, pandas.DataFrame):
        preds = preds.to_dict(orient="records")
    return jsonable_encoder({"predictions": preds})
    