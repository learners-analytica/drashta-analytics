from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import os
from service.test import machine_test

load_dotenv()

app = FastAPI()

print(os.getenv("ALLOW_ORIGINS"))
# Initialize Quick Data

app.add_middleware(
    CORSMiddleware,
    allow_origins = [str(os.getenv("ALLOW_ORIGINS"))],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/test/")
async def machine_test_api():
    data = machine_test()
    return jsonable_encoder(data)


