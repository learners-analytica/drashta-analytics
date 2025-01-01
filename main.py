from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.types.appTypes import MLModel
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Initialize Quick Data

ML_model = list[MLModel] = []

app.add_middleware(
    CORSMiddleware,
    allow_origins = [os.getenv("ALLOW_ORIGINS", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
