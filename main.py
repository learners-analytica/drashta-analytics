from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import dotenv
import httpx

from services.driver.model_query import TMLModelQuery,TModelPredictRequest, model_query_handle, model_predict_handle, get_model_list
from services.driver.model_db import get_model_list
from services.web.response import fetch_status, get_status_code


dotenv.load_dotenv()
app = FastAPI()


print(os.getenv("BRIDGE_SERVER"))

# Middleware for handling CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    url_bridge = os.getenv("BRIDGE_SERVER")
    url_client = os.getenv("CLIENT_SERVER")

    bridge_status = 404  # Default to 404 (Not Found)
    client_status = 404

    async with httpx.AsyncClient() as client:
        if url_bridge:
            try:
                response = await client.get(url_bridge, timeout=5)
                bridge_status = get_status_code(response)
            except httpx.RequestError:
                pass  # Keep bridge_status as 404

        if url_client:
            try:
                response = await client.get(url_client, timeout=5)
                client_status = get_status_code(response)
            except httpx.RequestError:
                pass  # Keep client_status as 404

    return {
        "Bridge Server Status": bridge_status,
        "Client Server Status": client_status
    }

@app.post("/model-gen-query")
async def requestQueryModel(body:TMLModelQuery):
    return await model_query_handle(
        body.table,
        body.x,
        body.y,
        body.model_name,
        body.size,
        body.task
    )

@app.get("/model_list")
async def req_model_list():
    return get_model_list()

@app.post("/run_predict_on_model")
async def run_predict_on_model(body:TModelPredictRequest):
    return await model_predict_handle(body.x, body.model_id)