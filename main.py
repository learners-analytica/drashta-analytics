from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import os
from service.test import machine_test
from service.auto_machine_learning import request_train_model_on_data

load_dotenv()

app = FastAPI()

print(os.getenv("ALLOW_ORIGINS"))
# Initialize Quick Data

app.add_middleware(
    CORSMiddleware,
    allow_origins = [str(os.getenv("CLIENT_SERVER"))],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def root():
    return {"message": "Analytics Service Connected"}


@app.post("/test/")
async def machine_test_api(param1: str, param2: int):
    # Use the parameters for processing
    data = machine_test()
    return jsonable_encoder(data)




@app.post("/train/")
async def train_model_api(data: dict, target_column: str, task_type: str, optimization_metric: str, time_limit: int, model_name: str, train_split: float, val_split: float, test_split: float):
    model = request_train_model_on_data(
        data=data,
        target_column=target_column,
        task_type=task_type,
        optimization_metric=optimization_metric,
        time_limit=time_limit,
        model_name=model_name,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split
    )
    return jsonable_encoder(model)
