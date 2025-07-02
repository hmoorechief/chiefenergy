# backend_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pyodbc
from datetime import date as dt_date
import os

app = FastAPI()

# Enable CORS so your Streamlit frontend can call this API from another origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read DB config from environment variables
server = os.getenv("SQL_SERVER", "dal-sql-02")
database = os.getenv("SQL_DB", "Enverus")
trusted_connection = os.getenv("SQL_TRUSTED_CONNECTION", "yes")
trust_cert = os.getenv("SQL_TRUST_CERTIFICATE", "yes")

def fetch_production_data_from_sql(api_list, start_date_filter, end_date_filter):
    if not api_list:
        return pd.DataFrame()

    try:
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 18 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'Trusted_Connection={trusted_connection};'
            f'TrustServerCertificate={trust_cert};'
        )

        placeholders = ','.join('?' for _ in api_list)
        query = f"""
            SELECT p.API_UWI_Unformatted AS API, CAST(p.ProducingMonth AS DATE) AS Date,
                   SUM(p.LiquidsProd_BBL) AS LiquidsProd_BBL, SUM(p.GasProd_MCF) AS GasProd_MCF,
                   SUM(p.WaterProd_BBL) AS WaterProd_BBL, w.Latitude, w.Longitude, p.WellID
            FROM dbo.production p
            LEFT JOIN dbo.well_headers w ON p.API_UWI_Unformatted = w.API_UWI_Unformatted
            WHERE p.API_UWI_Unformatted IN ({placeholders})
              AND p.ProducingMonth >= ?
              AND p.ProducingMonth <= ?
              AND p.LiquidsProd_BBL IS NOT NULL AND p.GasProd_MCF IS NOT NULL AND p.WaterProd_BBL IS NOT NULL
            GROUP BY p.API_UWI_Unformatted, p.ProducingMonth, w.Latitude, w.Longitude, p.WellID
            ORDER BY p.API_UWI_Unformatted, p.ProducingMonth
        """

        params = api_list + [start_date_filter.strftime('%Y-%m-%d'), end_date_filter.strftime('%Y-%m-%d')]
        df = pd.read_sql_query(query, conn, params=params)
        df['Date'] = pd.to_datetime(df['Date']).dt.to_period("M").dt.to_timestamp()
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if 'conn' in locals() and conn:
                conn.close()
        except:
            pass

@app.get("/production")
def get_production_data(apis: str, start: str, end: str):
    """
    Query parameters:
    - apis: comma-separated list of API numbers (e.g. "123,456,789")
    - start: start date in YYYY-MM-DD format
    - end: end date in YYYY-MM-DD format
    """
    api_list = [api.strip() for api in apis.split(",") if api.strip()]
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")

    df = fetch_production_data_from_sql(api_list, start_date, end_date)
    if df.empty:
        return {"message": "No data found for the given APIs and date range."}
    return df.to_dict(orient="records")

@app.get("/health")
def health_check():
    return {"status": "ok"}
