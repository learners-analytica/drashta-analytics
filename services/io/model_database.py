from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from drashta_types.drashta_types_model import TModelMetadata
from drashta_types.drashta_types_key import MLTaskTypes
import datetime
from pydantic import create_model
import dotenv
import os
import json

dotenv.load_dotenv()

class Model_DB_Fields(SQLModel, table=True):
    id: str = Field(primary_key=True)
    model_name: str = Field(index=True)
    data: str
    target: str
    date: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    task: MLTaskTypes
    estimator: str

db_file = os.environ("MODEL_DB_FILE")
db_url = f"sqlite:///{db_file}"

connect_args = {"check_same_thread": False}
engine = create_engine(db_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)