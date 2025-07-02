from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pyodbc

app = FastAPI()

server = 'dal-sql-02'
database = 'Enverus'

conn_str = (
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)

class RequestData(BaseModel):
    api_list: List[str]
    start_date: str
    end_date: str
    loe_values: dict

def fetch_production_data(api_list, start_date, end_date):
    if not api_list:
        return pd.DataFrame()
    try:
        conn = pyodbc.connect(conn_str)
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
        params = api_list + [start_date, end_date]
        df = pd.read_sql(query, conn, params=params)
        df['Date'] = pd.to_datetime(df['Date']).dt.to_period("M").dt.to_timestamp()
        return df
    finally:
        if 'conn' in locals():
            conn.close()

def load_hh_prices():
    url = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
    df = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
    col = "Henry Hub Natural Gas Spot Price (Dollars per Million Btu)"
    df.rename(columns={col: "GasPrice"}, inplace=True)
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    return df[["Date", "GasPrice"]]

def load_wti_prices():
    url = "https://www.eia.gov/dnav/pet/hist_xls/RWTCm.xls"
    df = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
    col = "Cushing, OK WTI Spot Price FOB (Dollars per Barrel)"
    df.rename(columns={col: "OilPrice"}, inplace=True)
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    return df[["Date", "OilPrice"]]

def calculate_economics(prod_df, gas_df, oil_df, loe_values):
    merged = prod_df.merge(gas_df, on="Date", how="left").merge(oil_df, on="Date", how="left")
    tax_adj = loe_values.get("Tax", 0.93) * 0.8
    merged["Liq_Econ"] = merged["LiquidsProd_BBL"] * merged["OilPrice"] * tax_adj
    merged["Liq_Exp"] = merged["LiquidsProd_BBL"] * loe_values.get("Liq", 1.5)
    merged["Gas_Econ"] = merged["GasProd_MCF"] * merged["GasPrice"] * tax_adj
    merged["Gas_Exp"] = merged["GasProd_MCF"] * loe_values.get("Gas", 1.5)
    merged["Water_Exp"] = merged["WaterProd_BBL"] * loe_values.get("Water", 0.5)
    merged["Extra_Expenses"] = loe_values.get("Expenses", 1000)
    merged["Total_Cost"] = (
        merged["Liq_Econ"] + merged["Gas_Econ"]
        - (merged["Liq_Exp"] + merged["Gas_Exp"] + merged["Water_Exp"] + merged["Extra_Expenses"])
    )
    merged["Econ_Flag"] = merged["Total_Cost"] >= 0
    return merged

@app.post("/calculate_economics")
def api_calculate_economics(req: RequestData):
    try:
        prod_df = fetch_production_data(req.api_list, req.start_date, req.end_date)
        gas_df = load_hh_prices()
        oil_df = load_wti_prices()
        econ_df = calculate_economics(prod_df, gas_df, oil_df, req.loe_values)
        return {"data": econ_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
