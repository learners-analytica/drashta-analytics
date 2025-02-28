from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from drashta_types.drashta_types_model import TModelMetadata
from drashta_types.drashta_types_key import MLTaskTypes
import datetime
from pydantic import create_model
import dotenv
import os

dotenv.load_dotenv()

class Model_DB_Fields(SQLModel, table=True):
    id: str = Field(primary_key=True)
    model_name: str = Field(index=True)
    data: str
    target: str
    date: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    task: MLTaskTypes
    estimator: str
    file_path:str

db_file = os.environ("MODEL_DB_FILE")
db_url = f"sqlite:///{db_file}"

connect_args = {"check_same_thread": False}
engine = create_engine(db_url, connect_args=connect_args)

SQLModel.metadata.create_all(engine)
    
def fetch_model_data(id:str)->Model_DB_Fields:
    with Session(engine) as session:
        statement = select(Model_DB_Fields).where(Model_DB_Fields.id == id)
        model_data = session.exec(statement).first()
        return model_data

def add_new_model(model_meta:TModelMetadata,file_name:str):
    model_entry = Model_DB_Fields(
        id = model_meta.id,
        model_name= ",".join(model_meta.data),
        data = model_meta.data,
        target= model_meta.target,
        task=model_meta.task,
        estimator=model_meta.estimator,
        file_path = file_name
    )
    with Session(engine) as session:
        session.add(model_entry)
        session.commit()
    
    
