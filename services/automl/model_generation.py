from flaml.automl import AutoML
from pandas import DataFrame

def generate_model(data: DataFrame, x_cols: list[str], y_cols: str, task: str, time_budget:int) -> AutoML:
    automl = AutoML()

    X_train = data[x_cols]
    y_train = data[y_cols]
    n_splits = len(X_train) if len(X_train) < 5 else 5

    automl.fit(
        X_train,
        y_train,
        task=task,
        time_budget=time_budget,
        n_splits=n_splits,
    )
    
    return automl


def get_best_model(model:AutoML)->AutoML:
    return model.model

def predict_model(model:any, data:DataFrame)->DataFrame:
    return model.predict(data)
