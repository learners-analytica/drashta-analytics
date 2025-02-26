from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import dotenv

from .services.driver.model_query import TMLModelQuery, MLModelQueryHandle

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

@app.post("/model-gen-query")
async def requestQueryModel(body:TMLModelQuery):
    MLModelQueryHandle(
        body.table,
        body.x,
        body.y,
        body.model_name,
        body.size,
        body.task
    )



