# Drashta Analytics

![Drashta Analytics Tech Stack](https://github-readme-tech-stack.vercel.app/api/cards?title=Drashta+Analytics+Tech+Stack&lineCount=3&line1=FLAML%2CFLAML%2CF4842D%3Bscikitlearn%2Cscikitlearn%2CF7931E%3B&line2=pydantic%2Cpydantic%2CE92063%3Bpandas%2Cpandas%2C150458%3B&line3=fastapi%2Cfastapi%2C009688%3BSQLModel%2CSQLModel%2C512BD4%3Bsqlite%2Csqlite%2C003B57%3B)


Service for Model Training for the Drashta Demo System running on FastAPI

# Features

1. AutoML Model Training : Trains Models using FLAML
2. Brdige Service Data Retrival : Retrives Data for model training using [Drashta Bridge](https://github.com/learners-analytica/drashta-bridge) and splits as per the request
3. Model Registry : Stores Model Files and hold metadata in a SQLite File including `model_name` , `columns` (input columns for model) , `target` (target column of model), `task` and `estimator`

# Packages 
* See `requirements.txt`
* Depends on a currently depriciated types library [`drashta-types-py`](https://github.com/learners-analytica/drashta-types-py)

# Running
1. Create a `Python 3.13` enviroment
2. run `pip install -r requirements.txt`
3. run `fastapi run main.py`
