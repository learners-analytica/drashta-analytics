from flaml import AutoML
from joblib import dump as joblib_dump
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from types.appTypes import MLModel

def _train_automl(
    data: DataFrame,
    target: str,
    time_budget: int = 60,
    metric: str = 'accuracy',
    task: str = None,
    train_test_split_size: float = 0.2,
    validation_size: float = 0.2,
) -> AutoML:
    """Trains an AutoML model on the given data and returns the model."""
    automl = AutoML()

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_split_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=42
    )

    automl.fit(
        X_train=X_train,
        y_train=y_train,
        time_budget=time_budget,
        metric=metric,
        task=task,
        X_val=X_val,
        y_val=y_val
    )

    predictions = automl.predict(X_test)
    accuracy = (predictions == y_test).mean()

    return automl

    