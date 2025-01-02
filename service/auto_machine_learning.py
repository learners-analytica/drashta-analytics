from flaml import AutoML
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from libapp.types import MLModelTable
import pickle

def request_train_model_on_data(
    data:DataFrame,
    target_column:str,
    task_type: str,
    optimization_metric: str,
    time_limit:int,
    model_name:str,  
    train_split:float,
    val_split:float,
    test_split:float
)->MLModelTable:
    file_path = f'./models/{model_name}.pkl'
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_split, test_size=val_split+test_split, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=val_split/(val_split+test_split), test_size=test_split/(val_split+test_split), random_state=42)
    
    automl = AutoML()
    automl.fit(X_train, y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, task=task_type, eval_metric=optimization_metric, time_limit=time_limit)
    with open(file_path, 'wb') as f:
        pickle.dump(automl, f)
        
    return MLModelTable(
        model_name=model_name,
        model_file_path=file_path,
        model_data_columns=list(X.columns),
        model_target=target_column,
        model_task_type=task_type,
        model_optimization_metric=optimization_metric,
        model_eval_metric_value=automl.best_score_,
        model_estimator_type=automl.best_estimator,
        model_eval_metric_info=automl.metrics_for_best_config
    )
        
    
