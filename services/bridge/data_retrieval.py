import httpx
import os
import dotenv
from drashta_types.drashta_types_table import TTableData
from drashta_types.drashta_types_data import TDataArray
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
        response:TTableData = await client.post(
            f"{bridge_server_url}/supabase/get-table-data-raw",
            json={"table": table, "columns": columns, "size": size}
        )
        
        response.raise_for_status()
        return response.json()  # Assuming response JSON can be mapped to TTableData
    
async def get_table_data(table:str,columns:list[str],size:int)->TDataArray:
    return (await request_table_data(table,columns,size))["table_data_series"]

async def get_table_dataframe(table:str,columns:list[str],size:int)->pandas.DataFrame:
    
    table_data:TDataArray = await get_table_data(table,columns,size)
    df = pandas.DataFrame(table_data)
    return df
