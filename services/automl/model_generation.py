from flaml.automl import AutoML
from pandas import DataFrame

def generate_model(data:DataFrame, x_cols:list[str], y_cols:str, task:str, train_time:float)->AutoML:
    automl = AutoML()
    automl.fit(
        X=data[x_cols],
        y=data[y_cols],
        task=task,
        train_time_limit=train_time
    )

def get_best_model(model:AutoML)->AutoML:
    return model.model

def predict_model(model:any, data:DataFrame)->DataFrame:
    return model.predict(data)
