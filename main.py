from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from drashta_types.drashta_types_data import TColumnNames
from drashta_types.drashta_types_table import TTableStructure
import httpx
import os
import dotenv
import pandas

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


@app.get("/get-database-struct")
async def get_database_struct()->list[TTableStructure]:
    bridge_server_url = os.getenv("BRIDGE_SERVER")
    
    if not bridge_server_url:
        raise HTTPException(status_code=500, detail="BRIDGE_SERVER environment variable is not set")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f'{bridge_server_url}/supabase/get-database-structure')

        response.raise_for_status()  # This will raise an exception for bad status codes (4xx, 5xx)
        print(response.json())
        return response.json()  # Assuming response JSON can be mapped to TColumnNames
    
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"Request error occurred: {exc}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"HTTP error occurred: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Error parsing response: {exc}")

