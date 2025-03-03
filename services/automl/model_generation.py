from flaml.automl import AutoML
from pandas import DataFrame

def generate_model(data: DataFrame, x_cols: list[str], y_cols: str, task: str) -> AutoML:
    automl = AutoML()

    X_train = data[x_cols]
    y_train = data[y_cols]

    automl.fit(
        X_train,
        y_train,
        task=task,
        time_budget=5
    )
    
    return automl


def get_best_model(model:AutoML)->AutoML:
    return model.model

def predict_model(model:any, data:DataFrame)->DataFrame:
    return model.predict(data)
