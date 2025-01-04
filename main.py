from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import os
from service.test import machine_test

load_dotenv()

app = FastAPI()

# Initialize Quick Data

app.add_middleware(
    CORSMiddleware,
    allow_origins = [os.getenv("ALLOW_ORIGINS", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test():
    data = machine_test()
    return jsonable_encoder(data)


