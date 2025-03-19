from typing import Annotated
from xml.parsers.expat import model
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from drashta_types.drashta_types_model import TModelMetadata
from drashta_types.drashta_types_key import MLTaskTypes
from drashta_types.drashta_types_data import TDataSeriesMinimal
import datetime
import json
from pydantic import create_model
import dotenv
import os

dotenv.load_dotenv()

class Model_DB_Fields(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    columns: str
    target: str
    date: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    task: MLTaskTypes
    estimator: str
    file_path:str

db_file = os.getenv("MODEL_DB_FILE")
db_url = f"sqlite:///{db_file}"

connect_args = {"check_same_thread": False}
engine = create_engine(db_url, connect_args=connect_args)

SQLModel.metadata.create_all(engine)

def safe_json_loads(data):
    """Ensure data is properly converted from string to JSON."""
    if isinstance(data, str):  # If it's a string, parse it
        return json.loads(data)
    return data  # If it's already a list/dict, return as is

def convert_to_tmodel_metadata(model_list):
    converted_list = []
    
    for model in model_list:
        converted_list.append(TModelMetadata(
            id=model.id,
            name=model.name,
            estimator=model.estimator,
            task=model.task.value,  # Assuming task is an Enum
            date=model.date.isoformat(),
            columns=[TDataSeriesMinimal(**col) for col in safe_json_loads(model.columns)],
            target=[TDataSeriesMinimal(**tgt) for tgt in safe_json_loads(model.target)]
        ))

    return converted_list
    
def fetch_model_list()->list[TModelMetadata]:
    with Session(engine) as session:
        model_data_stringed = session.exec(select(Model_DB_Fields)).all()
        model_list = convert_to_tmodel_metadata(model_data_stringed)
        return model_list

async def fetch_model_data(id:str)->Model_DB_Fields:
    with Session(engine) as session:
        statement = select(Model_DB_Fields).where(Model_DB_Fields.id == id)
        model_data_stringed = session.exec(statement).first()
        model_data = convert_to_tmodel_metadata([model_data_stringed])
        return model_data[0]

def add_new_model(model_meta:TModelMetadata,file_name:str):
    model_entry = Model_DB_Fields(
        id = model_meta.id,
        name= model_meta.name,
        columns = json.dumps(model_meta.model_dump()["columns"]),
        target= json.dumps(model_meta.model_dump()["target"]),
        task=model_meta.task,
        estimator=model_meta.estimator,
        file_path = file_name
    )
    with Session(engine) as session:
        session.add(model_entry)
        session.commit()
    
def remove_model(id:str)->bool:
    with Session(engine) as session:
        statement = select(Model_DB_Fields).where(Model_DB_Fields.id == id)
        model_data = session.exec(statement).first()
        if model_data:
            session.delete(model_data)
            session.commit()
            return True
        else:
            return False

async def fetch_model_filepath(id:str)->str:
    with Session(engine) as session:
        statement = select(Model_DB_Fields).where(Model_DB_Fields.id == id)
        model_data = session.exec(statement).first()
        return model_data.file_path
