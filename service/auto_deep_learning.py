import autokeras
from pandas import DataFrame
from sklearn.model_selection import train_test_split

def auto_text_regress(data:DataFrame, target_column:str, train_split:float, val_split:float, test_split:float, multi_label_check:bool)->autokeras.TextClassifier:
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_split, test_size=val_split+test_split, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=val_split/(val_split+test_split), test_size=test_split/(val_split+test_split), random_state=42)
    
    autoDlRegress = autokeras.TextClassifier(objective='val_accuracy', max_trials=10, multi_label=multi_label_check)
    autoDlRegress.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    autoDlRegress.evaluate(X_test, y_test)
    with open('./models/auto_dl_text_regressor.h5', 'wb') as f:
        autoDlRegress.export_model(f)
    return None