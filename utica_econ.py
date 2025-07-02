import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import io
import pyodbc
from datetime import date as dt_date  # Import date specifically for date_input defaults

# SQL Configuration
server = 'dal-sql-02'
database = 'Enverus'

# Ensure session states are initialized
if "last_api_hash" not in st.session_state:
    st.session_state.last_api_hash = None
# Rename to raw_production_data to store the unmerged data from SQL
if "raw_production_data" not in st.session_state:
    st.session_state.raw_production_data = None
# New session state for the fully calculated economic results
if "economic_results_df" not in st.session_state:
    st.session_state.economic_results_df = None


# --- Cached Function to Fetch Production Data from SQL ---
@st.cache_data(ttl=3600, show_spinner="Fetching production data from SQL...")
def fetch_production_data_from_sql(api_list, start_date_filter, end_date_filter):
    """
    Fetches production data for selected APIs and a specific date range from the SQL Server database.
    """
    if not api_list:
        st.warning("No APIs provided for SQL query.")
        return pd.DataFrame()  # Return empty if no APIs selected

    try:
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 18 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            'Trusted_Connection=yes;'
            'TrustServerCertificate=yes;'
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
              AND p.ProducingMonth <= ? -- Added end date filter
              AND p.LiquidsProd_BBL IS NOT NULL AND p.GasProd_MCF IS NOT NULL AND p.WaterProd_BBL IS NOT NULL
            GROUP BY p.API_UWI_Unformatted, p.ProducingMonth, w.Latitude, w.Longitude, p.WellID
            ORDER BY p.API_UWI_Unformatted, p.ProducingMonth
        """
        # Parameters for the SQL query: API list first, then start_date, then end_date
        params = api_list + [start_date_filter.strftime('%Y-%m-%d'), end_date_filter.strftime('%Y-%m-%d')]

        # Use pd.read_sql_query for robust parameter handling
        df = pd.read_sql_query(query, conn, params=params)

        # Ensure 'Date' column is datetime and then convert to monthly period timestamp
        df['Date'] = pd.to_datetime(df['Date']).dt.to_period("M").dt.to_timestamp()

        st.success(f"Successfully fetched {len(df)} rows for {len(api_list)} APIs.")
        return df
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        st.error(f"SQL database error: {ex}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during SQL data fetching: {e}")
        return pd.DataFrame()
    finally:
        try:
            if 'conn' in locals() and conn:
                conn.close()
        except:
            pass


# --- Cached Function to Load Henry Hub Prices ---
@st.cache_data(ttl=3600, show_spinner="Loading Henry Hub prices...")
def load_hh_prices():
    url = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
    try:
        df = pd.read_excel(url, sheet_name="Data 1", skiprows=2)

        original_col_name = "Henry Hub Natural Gas Spot Price (Dollars per Million Btu)"
        target_col_name = "GasPrice"

        if original_col_name in df.columns:
            df = df.rename(columns={original_col_name: target_col_name})
        elif len(df.columns) > 1:
            df = df.rename(columns={df.columns[1]: target_col_name})
        else:
            st.error(
                f"Henry Hub price data did not contain the expected column: '{original_col_name}' or a second column. "
                f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()

        df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df.dropna(subset=["Date"], inplace=True)
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
        return df[["Date", target_col_name]].sort_values("Date")
    except Exception as e:
        st.error(f"Failed to load Henry Hub prices from URL: {url}. Error: {e}")
        return pd.DataFrame()


# --- Cached Function to Load WTI Prices ---
@st.cache_data(ttl=3600, show_spinner="Loading WTI prices...")
def load_wti_prices():
    url = "https://www.eia.gov/dnav/pet/hist_xls/RWTCm.xls"
    try:
        df = pd.read_excel(url, sheet_name="Data 1", skiprows=2)

        original_col_name = "Cushing, OK WTI Spot Price FOB (Dollars per Barrel)"
        target_col_name = "OilPrice"

        if original_col_name in df.columns:
            df = df.rename(columns={original_col_name: target_col_name})
        elif len(df.columns) > 1:
            df = df.rename(columns={df.columns[1]: target_col_name})
        else:
            st.error(
                f"WTI oil price data did not contain the expected column: '{original_col_name}' or a second column. "
                f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()

        df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df.dropna(subset=["Date"], inplace=True)
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
        return df[["Date", target_col_name]].sort_values("Date")
    except Exception as e:
        st.error(f"Failed to load WTI oil prices from URL: {url}. Error: {e}")
        return pd.DataFrame()


# --- Cached Function for Economic Calculations ---
@st.cache_data(ttl=600, show_spinner="Calculating economics...")
def calculate_economics(prod_df, gas_df, oil_df, loe_values):
    GAS_PRICE_COL = "GasPrice"
    OIL_PRICE_COL = "OilPrice"

    # Merge raw production data with gas and oil prices
    # With `prod_df` consistently being the raw data, these merges should introduce 'GasPrice' and 'OilPrice'
    # without suffixes, as prod_df does not have these columns initially.
    merged = prod_df.merge(gas_df, on="Date", how="left") \
        .merge(oil_df, on="Date", how="left")

    # Now, check for the expected column names without suffixes
    if GAS_PRICE_COL not in merged.columns:
        st.error(f"Error in calculate_economics: '{GAS_PRICE_COL}' not found after merging. "
                 f"This indicates an issue with the gas price data or merge process.")
        st.write("Available columns in merged_df:", merged.columns.tolist())
        return pd.DataFrame()
    if OIL_PRICE_COL not in merged.columns:
        st.error(f"Error in calculate_economics: '{OIL_PRICE_COL}' not found after merging. "
                 f"This indicates an issue with the oil price data or merge process.")
        st.write("Available columns in merged_df:", merged.columns.tolist())
        return pd.DataFrame()

    # Convert price columns to numeric, handling potential errors and interpolating NaNs
    merged[GAS_PRICE_COL] = pd.to_numeric(merged[GAS_PRICE_COL], errors='coerce')
    merged[OIL_PRICE_COL] = pd.to_numeric(merged[OIL_PRICE_COL], errors='coerce')

    merged[GAS_PRICE_COL].interpolate(method="linear", inplace=True)
    merged[OIL_PRICE_COL].interpolate(method="linear", inplace=True)

    if merged[GAS_PRICE_COL].isnull().any():
        st.warning(f"Warning: '{GAS_PRICE_COL}' still contains NaN values after interpolation. "
                   "This may indicate missing historical price data for the selected period. Filling with 0.")
        merged[GAS_PRICE_COL].fillna(0, inplace=True)
    if merged[OIL_PRICE_COL].isnull().any():
        st.warning(f"Warning: '{OIL_PRICE_COL}' still contains NaN values after interpolation. "
                   "This may indicate missing historical price data for the selected period. Filling with 0.")
        merged[OIL_PRICE_COL].fillna(0, inplace=True)

    # Perform economic calculations using the correct price column names
    tax_adj = loe_values["Tax"] * 0.8
    merged["Liq_Econ"] = merged["LiquidsProd_BBL"] * merged[OIL_PRICE_COL] * tax_adj
    merged["Liq_Exp"] = merged["LiquidsProd_BBL"] * loe_values["Liq"]
    merged["Gas_Econ"] = merged["GasProd_MCF"] * merged[GAS_PRICE_COL] * tax_adj
    merged["Gas_Exp"] = merged["GasProd_MCF"] * loe_values["Gas"]
    merged["Water_Exp"] = merged["WaterProd_BBL"] * loe_values["Water"]
    merged["Extra_Expenses"] = loe_values["Expenses"]

    merged["Total_Cost"] = (
            merged["Liq_Econ"] + merged["Gas_Econ"]
            - (merged["Liq_Exp"] + merged["Gas_Exp"] + merged["Water_Exp"] + merged["Extra_Expenses"])
    )
    merged["Econ_Flag"] = merged["Total_Cost"] >= 0
    return merged


# --- Helper function for longest streak ---
def longest_streak(series):
    max_streak = streak = 0
    for val in series:
        streak = streak + 1 if val else 0
        max_streak = max(max_streak, streak)
    return max_streak


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Well Economics Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Economic Analyzer"])

if page == "Economic Analyzer":
    st.title("Automated Well Economics from API Numbers")
    api_input = st.text_area("Paste API Numbers (one per line, up to 500 supported for efficiency):")

    # LOE Parameters in Sidebar
    st.sidebar.header("LOE Parameters")
    loe_values = {
        'Liq': st.sidebar.number_input("Liq LOE ($/BBL)", value=1.5),
        'Gas': st.sidebar.number_input("Gas LOE ($/MCF)", value=1.5),
        'Water': st.sidebar.number_input("Water LOE ($/BBL)", value=0.5),
        'Expenses': st.sidebar.number_input("Fixed Monthly Expenses ($)", value=1000.0),
        'Tax': st.sidebar.number_input("Tax Multiplier", value=0.93)
    }

    # Date and Months to Analyze in Sidebar
    st.sidebar.header("Analysis Period")
    months = st.sidebar.slider("Months to Analyze", 6, 60, 24)

    current_date_dt = dt_date.today()
    start_year = st.sidebar.number_input("Analysis Start Year", min_value=2000, max_value=current_date_dt.year,
                                         value=2019)

    # Ensure start_date_filter and end_date_filter are pandas.Timestamp objects
    start_date_filter = pd.Timestamp(f"{start_year}-01-01")
    end_date_filter = start_date_filter + pd.DateOffset(months=months)

    if api_input:
        api_list = [x.strip() for x in api_input.splitlines() if x.strip()]
        if len(api_list) > 500:
            st.warning("You have entered more than 500 API numbers. Processing the first 500 for performance.")
            api_list = api_list[:500]

        # Hash with API list and date range to determine if SQL re-fetch is needed
        current_api_hash = hash((tuple(api_list), start_date_filter, end_date_filter))

        # Only fetch from SQL if APIs or date range changed
        if current_api_hash != st.session_state.last_api_hash:
            st.session_state.last_api_hash = current_api_hash
            # Store the raw data from SQL in its dedicated session state variable
            st.session_state.raw_production_data = fetch_production_data_from_sql(api_list, start_date_filter,
                                                                                  end_date_filter)

        # Always use the raw production data from session state for calculations
        raw_prod_data_for_calc = st.session_state.raw_production_data

        if raw_prod_data_for_calc.empty:
            st.warning(
                "No production data found for the selected APIs and date range. Please check API numbers or database connection/data availability.")
            st.stop()

        gas_prices = load_hh_prices()
        oil_prices = load_wti_prices()

        if gas_prices.empty:
            st.warning("Could not load Henry Hub price data. Please check connection to EIA or file formats.")
            st.stop()
        if oil_prices.empty:
            st.warning("Could not load WTI price data. Please check connection to EIA or file formats.")
            st.stop()

        # Perform economic calculations using the raw production data
        merged_df = calculate_economics(raw_prod_data_for_calc, gas_prices, oil_prices, loe_values)

        if merged_df.empty:
            st.error(
                "Economic calculations failed. This might be due to missing price data after merging or errors in calculation. Check previous error messages.")
            st.stop()

        # Store the fully calculated economic data in its dedicated session state variable
        st.session_state.economic_results_df = merged_df

        results = []
        # Use the economic_results_df for summary and charting
        if "WellID" not in st.session_state.economic_results_df.columns:
            st.error("WellID column not found in production data after economic calculation. Cannot summarize results.")
            st.stop()

        for well, group in st.session_state.economic_results_df.groupby("WellID"):
            last_n = group.sort_values("Date").tail(months)
            if "Econ_Flag" not in last_n.columns:
                st.warning(f"Econ_Flag not found for Well ID {well}. Skipping economic status calculation.")
                continue

            econ_flags = last_n["Econ_Flag"]
            non_econ_months = months - econ_flags.sum()
            total_cost_sum = round(last_n["Total_Cost"].sum(), 2) if "Total_Cost" in last_n.columns else 0.0

            results.append({
                "Well ID": well,
                "Status": "Economic" if non_econ_months == 0 else f"{non_econ_months} months not economic",
                "Non_Economic_Months": non_econ_months,
                "Total Cost": total_cost_sum,
                "Longest Economic Streak": longest_streak(econ_flags),
                "Latitude": group["Latitude"].iloc[0] if "Latitude" in group.columns else None,
                "Longitude": group["Longitude"].iloc[0] if "Longitude" in group.columns else None
            })
        df_summary = pd.DataFrame(results)
        st.session_state.econ_df = df_summary  # This seems to be the one you want to use for the summary table.

        # --- Filter wells for display (apply filters to economic_results_df) ---
        wells_to_exclude = st.multiselect("Hide Wells from All Charts:", options=df_summary["Well ID"].unique())
        df_summary_filtered = df_summary[~df_summary["Well ID"].isin(wells_to_exclude)].copy()
        # Filter the main economic results DataFrame for charting
        economic_results_df_filtered = st.session_state.economic_results_df[
            ~st.session_state.economic_results_df["WellID"].isin(wells_to_exclude)].copy()

        # --- Display Summary Table ---
        st.subheader("Well Economic Summary")
        if not df_summary_filtered.empty:
            st.dataframe(df_summary_filtered)
        else:
            st.info("No well economic summary data to display after filtering.")

        # --- Map Visualization ---
        if not df_summary_filtered.empty and not df_summary_filtered["Latitude"].isnull().all():
            st.subheader("Well Map (Economic Status & Heat Map)")
            map_center_lat = df_summary_filtered["Latitude"].mean()
            map_center_lon = df_summary_filtered["Longitude"].mean()
            m = folium.Map(
                location=[map_center_lat, map_center_lon],
                zoom_start=8,
                control_scale=True
            )
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
            folium.TileLayer("Stamen Terrain", name="Stamen Terrain",
                             attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap").add_to(m)
            folium.TileLayer("Stamen Toner", name="Stamen Toner",
                             attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap").add_to(m)

            econ_layer = folium.FeatureGroup(name="Well Economic Status")
            for _, row in df_summary_filtered.iterrows():
                if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                    color = "green" if row["Non_Economic_Months"] <= 6 else "orange" if row[
                                                                                            "Non_Economic_Months"] <= 12 else "red"
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=4,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        tooltip=(f"Well ID: {row['Well ID']}<br>"
                                 f"Non-Economic Months: {row['Non_Economic_Months']}")
                    ).add_to(econ_layer)
            econ_layer.add_to(m)

            heat_layer = folium.FeatureGroup(name="Production Heat Map")
            heat_data = df_summary_filtered[["Latitude", "Longitude"]].dropna().values.tolist()
            if heat_data:
                HeatMap(heat_data, radius=10, blur=7).add_to(heat_layer)
            heat_layer.add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=1200, height=700)
        else:
            st.info("No valid well coordinates available to display the map after filtering.")

        # --- Monthly Total Cost by Well Chart ---
        if not economic_results_df_filtered.empty and "Date" in economic_results_df_filtered.columns and "Total_Cost" in economic_results_df_filtered.columns:
            st.subheader("Monthly Cash Flow by Well")
            all_well_ids = economic_results_df_filtered["WellID"].unique().tolist()
            selected_wells_for_cost_chart = st.multiselect(
                "Select Wells for Monthly Cash Flow Chart:",
                options=all_well_ids,
                default=all_well_ids[0:min(5, len(all_well_ids))]
            )

            if selected_wells_for_cost_chart:
                cost_chart_data = economic_results_df_filtered[
                    economic_results_df_filtered["WellID"].isin(selected_wells_for_cost_chart)].copy()
                fig = px.line(cost_chart_data, x="Date", y="Total_Cost", color="WellID",
                              title="Monthly Cash Flow by Well")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select wells from the multiselect above to view their monthly total cost.")
        else:
            st.warning(
                "Skipping monthly total cost chart: Filtered data is empty or missing 'Date'/'Total_Cost' columns.")

        # --- Revenue vs Expenses by Well Chart ---
        required_cols_for_bar = ["Liq_Econ", "Gas_Econ", "Liq_Exp", "Gas_Exp", "Water_Exp", "Extra_Expenses"]
        if not economic_results_df_filtered.empty and all(
                col in economic_results_df_filtered.columns for col in required_cols_for_bar):
            st.subheader("Revenue vs Expenses by Well")
            # Aggregate for the bar chart
            grouped = economic_results_df_filtered.groupby("WellID").agg({
                "Liq_Econ": "sum", "Gas_Econ": "sum",
                "Liq_Exp": "sum", "Gas_Exp": "sum", "Water_Exp": "sum", "Extra_Expenses": "sum"
            }).reset_index()

            grouped["Revenue"] = grouped["Liq_Econ"] + grouped["Gas_Econ"]
            grouped["Expenses"] = grouped[["Liq_Exp", "Gas_Exp", "Water_Exp", "Extra_Expenses"]].sum(axis=1)

            # Prepare data for stacked bar chart (Revenue positive, Expenses negative)
            plot_data = pd.DataFrame({
                'WellID': grouped['WellID'],
                'Revenue': grouped['Revenue'],
                'Expenses': -grouped['Expenses']
            })

            plot_data_melted = plot_data.melt(id_vars='WellID', var_name='Type', value_name='Amount')

            bar_fig = px.bar(plot_data_melted.sort_values("WellID"),
                             x="WellID",
                             y="Amount",
                             color="Type",
                             barmode="relative",
                             title="Revenue vs Expenses by Well (Expenses Negative)",
                             labels={"Amount": "USD", "WellID": "Well"},
                             height=500)
            bar_fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.warning(
                "Skipping Revenue vs Expenses chart: Filtered data is empty or missing some economic calculation columns.")

        # --- Non-Economical Wells Distribution Chart ---
        if not df_summary_filtered.empty:
            st.subheader("Non-Economical Wells Distribution")
            bins = [-1, 0, 3, 6, months + 1]
            labels = ["Economic", "1-3 Months Non-Econ", "4-6 Months Non-Econ", f"7-{months} Months Non-Econ"]

            df_summary_filtered['Non_Econ_Category'] = pd.cut(
                df_summary_filtered['Non_Economic_Months'],
                bins=bins,
                labels=labels,
                right=True
            )

            category_counts = df_summary_filtered['Non_Econ_Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Number of Wells']
            category_counts['Category'] = pd.Categorical(
                category_counts['Category'],
                categories=labels,
                ordered=True
            )
            category_counts = category_counts.sort_values('Category')

            non_econ_chart = px.bar(
                category_counts,
                x='Category',
                y='Number of Wells',
                title='Distribution of Non-Economic Months per Well',
                labels={'Category': 'Non-Economic Months Category', 'Number of Wells': 'Number of Wells'},
                color='Category',
                height=400
            )
            st.plotly_chart(non_econ_chart, use_container_width=True)
        else:
            st.warning("Skipping Non-Economical Wells Distribution chart: No wells to display after filtering.")

        # --- Download Button ---
        towrite = io.BytesIO()
        df_summary_filtered.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button("Download Summary as Excel", towrite.getvalue(), "Economic_Summary.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Paste API numbers in the text area to begin the economic analysis.")