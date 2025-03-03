import httpx
import os
import dotenv
from drashta_types.drashta_types_table import TTableData, TTableStructure, TDataArray
from drashta_types.drashta_types_data import TDataSeriesHead
import pandas
from pydantic import BaseModel
from typing import List

dotenv.load_dotenv()

class requestBodyRetrieveTableData(BaseModel):
    table: str
    columns: List[str]
    size: int

async def request_table_data(table: str, columns: list[str], size: int) -> TTableData:
    bridge_server_url = os.getenv("BRIDGE_SERVER")
    
    if not bridge_server_url:
        raise RuntimeError("BRIDGE_SERVER environment variable is not set")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{bridge_server_url}/supabase/get-table-data-raw",
            json={"table": table, "columns": columns, "size": size}
        )

        response.raise_for_status()  # Ensures request was successful

        return response.json()  # Correct way to parse JSON from HTTP response

async def request_table_column_data(table:str)->TTableStructure:
    bridge_server_url = os.getenv("BRIDGE_SERVER")
    if not bridge_server_url:
        raise RuntimeError("BRIDGE_SERVER environment variable is not set")
    async with httpx.AsyncClient() as client:
        response:list[TTableStructure] = await client.post(
            f"{bridge_server_url}/supabase/get-database-structure"
        )
    for i in response:
        if i.table_name == table:
            return i

async def request_column_data(table:str,columns:list[str])->list[TDataSeriesHead]:
    column_head_data:list[TDataSeriesHead] = []
    table_data:TTableStructure = request_table_column_data(table)
    table_columns:list[TDataSeriesHead] = table_data.table_column_head_data
    for col in table_columns:
        if col.column_name in columns:
            column_head_data.append(col)
    return column_head_data
    
async def get_table_data(table:str,columns:list[str],size:int)->TDataArray:
    return (await request_table_data(table,columns,size))["table_data_series"]

async def get_table_dataframe(table:str,columns:list[str],size:int)->pandas.DataFrame:
    table_data:TDataArray = await get_table_data(table,columns,size)
    df = pandas.DataFrame(table_data)
    return df