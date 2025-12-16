from fastapi import FastAPI, Depends, HTTPException, Security, Query
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_401_UNAUTHORIZED
from dotenv import load_dotenv
import os, json

# Load environment variables
load_dotenv()

app = FastAPI(title="Secure Stock Data API", version="2.0")

# --- Security Setup ---
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
):
    if api_key_header == API_KEY:
        return api_key_header
    elif api_key_query == API_KEY:
        return api_key_query
    else:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


# --- API Endpoint ---
@app.get('/')
async def frontpage():
    return {
        "message": "write /secure-data to enter into the main endpoint",
        "message1":"http://127.0.0.1:8000/secure-data?api_key=12345securekey67890&symbol=&sector=Banking&limit=5&sort_by=price&order=desc"
            }





@app.get("/secure-data")
async def get_data(
    api_key: str = Depends(get_api_key),
    symbol: str | None = Query(None, description="Filter by stock symbol (e.g., INFY)"),
    sector: str | None = Query(None, description="Filter by sector (e.g., Banking, Energy)"),
    limit: int = Query(10, ge=1, le=100, description="Limit number of results (1–100)"),
    sort_by: str | None = Query(None, description="Sort field: price, volume, or market_cap"),
    order: str = Query("asc", regex="^(asc|desc)$", description="Sort order: asc or desc"),
):
    """
    Retrieve secure stock data.
    Requires valid API key. Supports filtering, sorting, and limiting results.
    """
    with open("data.json", "r") as f:
        data = json.load(f)["stocks"]

    # Filter by symbol
    if symbol:
        data = [stock for stock in data if stock["symbol"].lower() == symbol.lower()]

    # Filter by sector
    if sector:
        data = [stock for stock in data if stock["sector"].lower() == sector.lower()]

    # Sort results
    if sort_by and sort_by in ["price", "volume", "market_cap"]:
        reverse = order == "desc"
        data = sorted(data, key=lambda x: x[sort_by], reverse=reverse)

    # Apply limit
    data = data[:limit]

    return {"status": "success", "count": len(data), "data": data}
