from sklearn.datasets import fetch_california_housing
import pandas as pd
from service.auto_machine_learning import request_train_model_on_data
from service.auto_machine_learning import request_predict_with_model


def machine_test():
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    print(data.tail())
    
    model = request_train_model_on_data(data, 'target', 'classification', 'accuracy', 10, 'iris', 0.8, 0.1, 0.1)
    print(model.model_file_path)
    
    last_row = data.iloc[-1][:-1].to_dict()
    print(last_row)
    print(data['target'].tail())

    pred = request_predict_with_model(model.model_file_path, pd.DataFrame(last_row, index=[0]))
    return {
        "statusCode": 200,
        "body": {
            "status": "success",
            "prediction": pred.tolist(),
            'model_type': model.model_estimator_type
        }
    }
