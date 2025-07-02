import streamlit as st
import pandas as pd
import io
import requests
from datetime import date as dt_date
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px

API_URL = "http://localhost:8000/calculate_economics"  # Change to your backend URL

# Your helper functions
def longest_streak(series):
    max_streak = streak = 0
    for val in series:
        streak = streak + 1 if val else 0
        max_streak = max(max_streak, streak)
    return max_streak

st.set_page_config(page_title="Well Economics Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Economic Analyzer"])

if page == "Economic Analyzer":
    st.title("Automated Well Economics from API Numbers")
    api_input = st.text_area("Paste API Numbers (one per line, up to 500 supported for efficiency):")

    st.sidebar.header("LOE Parameters")
    loe_values = {
        'Liq': st.sidebar.number_input("Liq LOE ($/BBL)", value=1.5),
        'Gas': st.sidebar.number_input("Gas LOE ($/MCF)", value=1.5),
        'Water': st.sidebar.number_input("Water LOE ($/BBL)", value=0.5),
        'Expenses': st.sidebar.number_input("Fixed Monthly Expenses ($)", value=1000.0),
        'Tax': st.sidebar.number_input("Tax Multiplier", value=0.93)
    }

    st.sidebar.header("Analysis Period")
    months = st.sidebar.slider("Months to Analyze", 6, 60, 24)
    current_date_dt = dt_date.today()
    start_year = st.sidebar.number_input("Analysis Start Year", min_value=2000, max_value=current_date_dt.year,
                                         value=2019)

    start_date_filter = f"{start_year}-01-01"
    end_year = start_year + (months // 12) + 1
    end_date_filter = f"{end_year}-12-31"

    if api_input:
        api_list = [x.strip() for x in api_input.splitlines() if x.strip()]
        if len(api_list) > 500:
            st.warning("You have entered more than 500 API numbers. Processing the first 500 for performance.")
            api_list = api_list[:500]

        payload = {
            "api_list": api_list,
            "start_date": start_date_filter,
            "end_date": end_date_filter,
            "loe_values": loe_values
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    st.warning("No production or economic data returned from backend.")
                    st.stop()
                df = pd.DataFrame(data)

                # Your original economic summary & charts logic here using df

                results = []
                for well, group in df.groupby("WellID"):
                    last_n = group.sort_values("Date").tail(months)
                    econ_flags = last_n["Econ_Flag"]
                    non_econ_months = months - econ_flags.sum()
                    total_cost_sum = round(last_n["Total_Cost"].sum(), 2)

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

                # Filtering wells
                wells_to_exclude = st.multiselect("Hide Wells from All Charts:", options=df_summary["Well ID"].unique())
                df_summary_filtered = df_summary[~df_summary["Well ID"].isin(wells_to_exclude)].copy()
                economic_results_df_filtered = df[~df["WellID"].isin(wells_to_exclude)].copy()

                # Display Summary Table
                st.subheader("Well Economic Summary")
                st.dataframe(df_summary_filtered)

                # Map Visualization (same as your original)
                if not df_summary_filtered.empty and not df_summary_filtered["Latitude"].isnull().all():
                    st.subheader("Well Map (Economic Status & Heat Map)")
                    map_center_lat = df_summary_filtered["Latitude"].mean()
                    map_center_lon = df_summary_filtered["Longitude"].mean()
                    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=8, control_scale=True)
                    folium.TileLayer("OpenStreetMap").add_to(m)
                    folium.TileLayer("Stamen Terrain").add_to(m)
                    folium.TileLayer("Stamen Toner").add_to(m)

                    econ_layer = folium.FeatureGroup(name="Well Economic Status")
                    for _, row in df_summary_filtered.iterrows():
                        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                            color = (
                                "green" if row["Non_Economic_Months"] <= 6
                                else "orange" if row["Non_Economic_Months"] <= 12
                                else "red"
                            )
                            folium.CircleMarker(
                                location=[row["Latitude"], row["Longitude"]],
                                radius=4,
                                color=color,
                                fill=True,
                                fill_opacity=0.7,
                                tooltip=(f"Well ID: {row['Well ID']}<br>Non-Economic Months: {row['Non_Economic_Months']}")
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

                # Monthly Total Cost by Well Chart
                if not economic_results_df_filtered.empty:
                    st.subheader("Monthly Cash Flow by Well")
                    all_well_ids = economic_results_df_filtered["WellID"].unique().tolist()
                    selected_wells_for_cost_chart = st.multiselect(
                        "Select Wells for Monthly Cash Flow Chart:",
                        options=all_well_ids,
                        default=all_well_ids[:min(5, len(all_well_ids))]
                    )
                    if selected_wells_for_cost_chart:
                        cost_chart_data = economic_results_df_filtered[
                            economic_results_df_filtered["WellID"].isin(selected_wells_for_cost_chart)
                        ].copy()
                        fig = px.line(cost_chart_data, x="Date", y="Total_Cost", color="WellID",
                                      title="Monthly Cash Flow by Well")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Select wells from the multiselect above to view their monthly total cost.")
                else:
                    st.warning("Skipping monthly total cost chart: No data after filtering.")

                # Revenue vs Expenses chart (same as original)
                required_cols = ["Liq_Econ", "Gas_Econ", "Liq_Exp", "Gas_Exp", "Water_Exp", "Extra_Expenses"]
                if not economic_results_df_filtered.empty and all(c in economic_results_df_filtered.columns for c in required_cols):
                    st.subheader("Revenue vs Expenses by Well")
                    grouped = economic_results_df_filtered.groupby("WellID").agg({
                        "Liq_Econ": "sum", "Gas_Econ": "sum",
                        "Liq_Exp": "sum", "Gas_Exp": "sum", "Water_Exp": "sum", "Extra_Expenses": "sum"
                    }).reset_index()
                    grouped["Revenue"] = grouped["Liq_Econ"] + grouped["Gas_Econ"]
                    grouped["Expenses"] = grouped[["Liq_Exp", "Gas_Exp", "Water_Exp", "Extra_Expenses"]].sum(axis=1)

                    plot_data = pd.DataFrame({
                        'WellID': grouped['WellID'],
                        'Revenue': grouped['Revenue'],
                        'Expenses': -grouped['Expenses']
                    })
                    plot_data_melted = plot_data.melt(id_vars='WellID', var_name='Type', value_name='Amount')

                    bar_fig = px.bar(
                        plot_data_melted.sort_values("WellID"),
                        x="WellID", y="Amount", color="Type",
                        barmode="relative",
                        title="Revenue vs Expenses by Well (Expenses Negative)",
                        labels={"Amount": "USD", "WellID": "Well"},
                        height=500
                    )
                    bar_fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(bar_fig, use_container_width=True)
                else:
                    st.warning("Skipping Revenue vs Expenses chart: Missing required columns or empty data.")

                # Non-Economical Wells Distribution Chart
                st.subheader("Non-Economical Wells Distribution")
                bins = [-1, 0, 3, 6, months + 1]
                labels = ["Economic", "1-3 Months Non-Econ", "4-6 Months Non-Econ", f"7-{months} Months Non-Econ"]
                df_summary_filtered['Non_Econ_Category'] = pd.cut(
                    df_summary_filtered['Non_Economic_Months'], bins=bins, labels=labels, right=True
                )
                category_counts = df_summary_filtered['Non_Econ_Category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Number of Wells']
                category_counts['Category'] = pd.Categorical(category_counts['Category'], categories=labels, ordered=True)
                category_counts = category_counts.sort_values('Category')

                non_econ_chart = px.bar(
                    category_counts, x='Category', y='Number of Wells',
                    title='Distribution of Non-Economic Months per Well',
                    labels={'Category': 'Non-Economic Months Category', 'Number of Wells': 'Number of Wells'},
                    color='Category', height=400
                )
                st.plotly_chart(non_econ_chart, use_container_width=True)

                # Download Button
                towrite = io.BytesIO()
                df_summary_filtered.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                st.download_button("Download Summary as Excel", towrite.getvalue(),
                                   "Economic_Summary.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            else:
                st.error(f"API call failed with status {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Error calling backend API: {e}")
    else:
        st.info("Paste API numbers in the text area to begin the economic analysis.")
