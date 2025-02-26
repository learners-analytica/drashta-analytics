from fastapi import Request
import httpx

def get_status_code(response: httpx.Response | None) -> int:
    return response.status_code if response else 404  # Default to 404 if response is None

async def fetch_status(url: str) -> str:
    """Fetch status code from a given URL, handling errors."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5)
            return f"{response.status_code} {response.reason_phrase}"
    except httpx.RequestError:
        return "Error: Unable to connect"
    except Exception as e:
        return f"Error: {str(e)}"

