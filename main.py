import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re # For parsing monthly pollutant column names

# --- Page Configuration ---
st.set_page_config(
    page_title="US Disease & Pollutant Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
HOTSPOT_DATA_PATH = "all_years_gnn_predictions_semi_supervised.csv"
POLLUTANT_DATA_PATH = "all_pollutants_merged_inner.csv" # Path to your pollutant data
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# --- Data Loading and Caching ---
@st.cache_data
def load_and_merge_data(hotspot_file_path: str, pollutant_file_path: str) -> pd.DataFrame:
    """
    Loads, cleans, preprocesses, and merges hotspot and pollutant data.
    Also identifies potential monthly pollutant columns.
    """
    if not os.path.exists(hotspot_file_path):
        st.error(f"Error: Hotspot data file '{hotspot_file_path}' not found.")
        return pd.DataFrame()
    if not os.path.exists(pollutant_file_path):
        st.error(f"Error: Pollutant data file '{pollutant_file_path}' not found.")
        return pd.DataFrame()

    try:
        # Load hotspot data
        df_hotspot = pd.read_csv(hotspot_file_path)
        # Corrected renaming based on the error message for hotspot data
        df_hotspot.rename(columns={
            "State Name": "State",  # Corrected based on error log
            "County Name": "County", # Corrected based on error log
            "Predicted_Latitude": "Latitude", 
            "Predicted_Hotspot_GNN": "Hotspot_Prediction_Score"
        }, inplace=True)

        if 'Year' not in df_hotspot.columns:
            st.error(f"Critical Error: 'Year' column not found in hotspot data file: '{hotspot_file_path}'. Columns found: {list(df_hotspot.columns)}")
            return pd.DataFrame()
        df_hotspot["Year"] = pd.to_numeric(df_hotspot["Year"], errors='coerce').astype("Int64")

        if "Hotspot_Prediction_Score" in df_hotspot.columns:
            df_hotspot["Hotspot_Prediction_Score"] = pd.to_numeric(df_hotspot["Hotspot_Prediction_Score"], errors='coerce')
        else:
            st.warning("Column 'Predicted_Hotspot_GNN' (for Hotspot_Prediction_Score) not found in hotspot data. Map color intensity may be uniform.")
            df_hotspot["Hotspot_Prediction_Score"] = 0.0

        if "Hotspot" in df_hotspot.columns:
            df_hotspot["Hotspot"] = pd.to_numeric(df_hotspot["Hotspot"], errors='coerce').fillna(0).astype(int)
        elif "Hotspot_Prediction_Score" in df_hotspot.columns:
            st.info("Deriving 'Hotspot' status from 'Hotspot_Prediction_Score' as 'Hotspot' column was not found in hotspot data.")
            df_hotspot["Hotspot"] = (df_hotspot["Hotspot_Prediction_Score"] > 0.5).astype(int)
        else:
            st.warning("Could not find or derive 'Hotspot' column in hotspot data. Defaulting to 0.")
            df_hotspot["Hotspot"] = 0
        
        if 'Latitude' in df_hotspot.columns:
            df_hotspot['Latitude'] = pd.to_numeric(df_hotspot['Latitude'], errors='coerce')
        if 'Longitude' in df_hotspot.columns:
            df_hotspot['Longitude'] = pd.to_numeric(df_hotspot['Longitude'], errors='coerce')

        # Load pollutant data
        df_pollutant = pd.read_csv(pollutant_file_path)
        
        if 'Date Local' in df_pollutant.columns:
            try:
                df_pollutant['Year'] = pd.to_datetime(df_pollutant['Date Local'], errors='coerce').dt.year
                df_pollutant['Year'] = pd.to_numeric(df_pollutant['Year'], errors='coerce').astype('Int64') 
                if df_pollutant['Year'].isna().all(): 
                    st.error(f"Critical Error: Could not extract valid years from 'Date Local' column in pollutant data file: '{pollutant_file_path}'. Please check date format.")
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Error processing 'Date Local' column in pollutant data: {e}. Columns found: {list(df_pollutant.columns)}")
                return pd.DataFrame()
        else:
            st.error(f"Critical Error: 'Date Local' column (needed to extract Year) not found in pollutant data file: '{pollutant_file_path}'. Columns found: {list(df_pollutant.columns)}")
            return pd.DataFrame()

        state_col_found_pollutant = False
        for potential_state_col in ['State', 'State Name', 'state', 'state_name']: 
            if potential_state_col in df_pollutant.columns:
                if potential_state_col != 'State':
                    df_pollutant.rename(columns={potential_state_col: 'State'}, inplace=True)
                state_col_found_pollutant = True
                break
        if not state_col_found_pollutant:
            st.warning(f"Warning: 'State' column (or common variations) not found in pollutant data file: '{pollutant_file_path}'. Merge might be incomplete. Columns found: {list(df_pollutant.columns)}")
        
        county_col_found_pollutant = False
        for potential_county_col in ['County', 'County Name', 'county', 'county_name']: 
            if potential_county_col in df_pollutant.columns:
                if potential_county_col != 'County':
                    df_pollutant.rename(columns={potential_county_col: 'County'}, inplace=True)
                county_col_found_pollutant = True
                break
        if not county_col_found_pollutant:
             st.warning(f"Warning: 'County' column (or common variations) not found in pollutant data file: '{pollutant_file_path}'. Merge might be incomplete. Columns found: {list(df_pollutant.columns)}")

        potential_pollutant_cols = [col for col in df_pollutant.columns if col not in ['Year', 'Date Local', 'State', 'County', 'FIPS', 'Latitude', 'Longitude', 'Site Num', 'State Code', 'County Code'] and ('CODE' not in col.upper())]
        numeric_pollutant_cols = []
        monthly_pollutant_groups = {}
        month_pattern = re.compile(r"(.+)_(" + "|".join(MONTH_ORDER) + r")$", re.IGNORECASE)

        for col in potential_pollutant_cols:
            try:
                df_pollutant[col] = pd.to_numeric(df_pollutant[col], errors='coerce')
                if df_pollutant[col].notna().sum() > 0 and df_pollutant[col].dtype in ['float64', 'int64']:
                    numeric_pollutant_cols.append(col)
                    match = month_pattern.match(col)
                    if match:
                        base_name = match.group(1)
                        if base_name not in monthly_pollutant_groups:
                            monthly_pollutant_groups[base_name] = []
                        monthly_pollutant_groups[base_name].append(col)
            except Exception: pass 
        
        for base_name in monthly_pollutant_groups:
            monthly_pollutant_groups[base_name].sort(key=lambda m_col: MONTH_ORDER.index(month_pattern.match(m_col).group(2).capitalize()))

        st.session_state['pollutant_columns'] = sorted(numeric_pollutant_cols)
        st.session_state['monthly_pollutant_groups'] = monthly_pollutant_groups
        
        required_keys = ["Year", "State", "County"]
        missing_keys_hotspot = [key for key in required_keys if key not in df_hotspot.columns]
        if missing_keys_hotspot:
            st.error(f"Hotspot data is missing required columns for merge: {missing_keys_hotspot} after renaming. Please check '{hotspot_file_path}'. Available columns: {list(df_hotspot.columns)}")
            return pd.DataFrame()

        missing_keys_pollutant = [key for key in required_keys if key not in df_pollutant.columns]
        if missing_keys_pollutant:
            st.error(f"Pollutant data is missing required columns for merge: {missing_keys_pollutant} after renaming. Please check '{pollutant_file_path}'. Available columns: {list(df_pollutant.columns)}")
            return pd.DataFrame()
        
        if 'Date Local' in df_pollutant.columns and len(df_pollutant[['Year', 'State', 'County', 'Date Local']].drop_duplicates()) > len(df_pollutant[['Year', 'State', 'County']].drop_duplicates()):
            st.info("Pollutant data appears to be at a sub-annual level. Aggregating to annual means for merging.")
            agg_cols = required_keys + [col for col in numeric_pollutant_cols if col in df_pollutant.columns]
            # Ensure all columns in agg_cols actually exist in df_pollutant before grouping
            agg_cols_present = [col for col in agg_cols if col in df_pollutant.columns]
            df_pollutant_agg = df_pollutant[agg_cols_present].groupby(required_keys).mean().reset_index()
            df_merged = pd.merge(df_hotspot, df_pollutant_agg, on=required_keys, how="left", suffixes=('', '_pollutant_dup'))
        else:
            df_merged = pd.merge(df_hotspot, df_pollutant, on=required_keys, how="left", suffixes=('', '_pollutant_dup'))
        
        cols_to_drop_from_merge = [col for col in df_merged.columns if '_pollutant_dup' in col]
        df_merged.drop(columns=cols_to_drop_from_merge, inplace=True, errors='ignore')

        critical_hotspot_cols_for_dropna = ["Year", "State", "County", "Latitude", "Longitude", "Hotspot", "Hotspot_Prediction_Score"]
        df_merged.dropna(subset=critical_hotspot_cols_for_dropna, inplace=True)

        return df_merged

    except Exception as e:
        st.error(f"An error occurred during data loading or merging: {e}")
        return pd.DataFrame()


# --- Helper Functions ---
def get_filtered_data(df: pd.DataFrame, selected_year: int, selected_state: str, selected_counties: list) -> pd.DataFrame:
    data_filtered = df[df["Year"] == selected_year]
    if selected_state != "All States":
        data_filtered = data_filtered[data_filtered["State"] == selected_state]
    if selected_counties: 
        data_filtered = data_filtered[data_filtered["County"].isin(selected_counties)]
    return data_filtered

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False).encode('utf-8')

# --- Main UI Rendering ---
def main():
    st.markdown("""
        <h1 style='text-align: center; color: #E63946;'>🌡️ US Disease Hotspot & Pollutant Dashboard</h1>
        <h3 style='text-align: center; color: #457B9D;'>Track, Analyze, and Predict Disease Hotspots with Pollutant Context</h3>
        <hr style='border:1px solid #1D3557'>
    """, unsafe_allow_html=True)

    df_full = load_and_merge_data(HOTSPOT_DATA_PATH, POLLUTANT_DATA_PATH)

    if df_full.empty:
        st.warning("No data loaded. Please check data files and try again. Ensure 'Year', 'State', and 'County' columns (or their common variations) exist in both CSVs.")
        return

    st.sidebar.header("🔍 Filters")
    available_years = sorted(df_full["Year"].dropna().unique().astype(int), reverse=True)
    if not available_years:
        st.sidebar.warning("No valid year data found."); return
    selected_year = st.sidebar.selectbox("Select Year", available_years, index=0) 

    state_options = ["All States"] + sorted(df_full["State"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", state_options)

    county_options_df = df_full[df_full["Year"] == selected_year]
    if selected_state != "All States":
        county_options_df = county_options_df[county_options_df["State"] == selected_state]
    county_options = sorted(county_options_df["County"].dropna().unique().tolist())
    
    selected_counties = []
    if county_options: 
        selected_counties = st.sidebar.multiselect("Select Counties (optional)", county_options, placeholder="All Counties in Selection")
    else:
        st.sidebar.info(f"No counties found for {selected_state} in {selected_year}.")

    pollutant_columns = st.session_state.get('pollutant_columns', [])
    selected_pollutant_sidebar = None
    if pollutant_columns:
        selected_pollutant_sidebar = st.sidebar.selectbox("Select Pollutant for General Analysis", pollutant_columns, index=0 if pollutant_columns else -1, key="selected_pollutant_sidebar")
    else:
        st.sidebar.info("No pollutant data columns found or loaded for analysis.")

    data_filtered = get_filtered_data(df_full, selected_year, selected_state, selected_counties)

    st.header(f"🔬 Analysis for {selected_year}")
    scope_name = selected_state
    if selected_counties: scope_name = f"{len(selected_counties)} selected counties in {selected_state}" if selected_state != "All States" else f"{len(selected_counties)} selected counties"
    elif selected_state == "All States": scope_name = "the entire US"
    st.markdown(f"*Displaying data for **{scope_name}***")

    if not data_filtered.empty:
        total_counties_in_scope = data_filtered["County"].nunique()
        hotspot_counties_current_year = data_filtered[data_filtered["Hotspot"] == 1]["County"].nunique()
        percentage_hotspots = (hotspot_counties_current_year / total_counties_in_scope * 100) if total_counties_in_scope > 0 else 0
        hotspot_counties_previous_year = 0
        if len(available_years) > 1 and selected_year > min(available_years): # Check if there are multiple years and selected is not the min
            data_previous_year = get_filtered_data(df_full, selected_year - 1, selected_state, selected_counties)
            if not data_previous_year.empty:
                 hotspot_counties_previous_year = data_previous_year[data_previous_year["Hotspot"] == 1]["County"].nunique()
        change_from_previous_year = hotspot_counties_current_year - hotspot_counties_previous_year
        
        kpi_cols = st.columns(4)
        with kpi_cols[0]: st.metric("Hotspot Counties", hotspot_counties_current_year)
        with kpi_cols[1]: st.metric("Total Counties in Scope", total_counties_in_scope)
        with kpi_cols[2]: st.metric("% Hotspot Counties", f"{percentage_hotspots:.1f}%")
        with kpi_cols[3]: st.metric(f"Change from {selected_year-1}", f"{change_from_previous_year:+.0f} counties", help="Compared to the same scope in the previous year.")
    else:
        st.info("No data available for the selected filters to display metrics.")
    st.markdown("---")

    tab_titles = ["🗺️ Hotspot Intensity Map", "📈 Trends & Predictions", "🏆 Top Counties"]
    if pollutant_columns: 
        tab_titles.append("🏭 Pollutant Analysis")
    tab_titles.append("💾 Data View & Download")
    
    tabs = st.tabs(tab_titles)
    
    map_tab = tabs[0]
    trend_tab = tabs[1]
    top_counties_tab = tabs[2]
    
    current_tab_index = 3
    pollutant_analysis_tab = None
    if pollutant_columns:
        pollutant_analysis_tab = tabs[current_tab_index]
        current_tab_index += 1
    data_view_tab = tabs[current_tab_index]

    with map_tab:
        st.subheader(f"🌡️ County-Level Hotspot Intensity for {selected_year}")
        if not data_filtered.empty and 'Latitude' in data_filtered.columns and 'Longitude' in data_filtered.columns:
            map_plot_data = data_filtered.dropna(subset=['Latitude', 'Longitude']).copy()
            if not map_plot_data.empty:
                map_plot_data['map_marker_visual_size'] = map_plot_data.get('Hotspot_Prediction_Score', 0.0) + (map_plot_data['Hotspot'] * 0.5)
                hover_data_map = {"State": True, "Hotspot_Prediction_Score": ":.2f", "Hotspot": True, "map_marker_visual_size": False, "Latitude": False, "Longitude": False}
                if selected_pollutant_sidebar and selected_pollutant_sidebar in map_plot_data.columns: 
                    hover_data_map[selected_pollutant_sidebar] = ":.2f" 
                fig_map = px.scatter_mapbox(
                    map_plot_data, lat='Latitude', lon='Longitude', color='Hotspot_Prediction_Score', 
                    size='map_marker_visual_size', color_continuous_scale=px.colors.sequential.YlOrRd, 
                    size_max=18, opacity=0.7, hover_name="County", hover_data=hover_data_map,
                    title=f"Hotspot Prediction Intensity in {scope_name} ({selected_year})",
                    zoom=3.5, center={"lat": 39.8283, "lon": -98.5795}, mapbox_style="carto-positron")
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, coloraxis_colorbar=dict(title="Prediction Score"))
                st.plotly_chart(fig_map, use_container_width=True)
            else: st.info("No data with valid coordinates to display on map.")
        else: st.info("Insufficient data for map (missing coordinates or filtered data is empty).")

    with trend_tab: 
        st.subheader("📈 Hotspot Count Trend & Prediction")
        relevant_counties_for_trend = data_filtered["County"].unique()
        if relevant_counties_for_trend.size > 0 :
            trend_data_df = df_full[df_full["County"].isin(relevant_counties_for_trend)] 
            trend_agg = (
                trend_data_df.groupby("Year")["Hotspot"].sum()
                .reindex(range(min(available_years) -1 if available_years else 2018, max(available_years) + 3 if available_years else 2026), fill_value=0) 
                .reset_index().rename(columns={"Hotspot": "Hotspot_Count"})
            )
            last_actual_data_year = df_full["Year"].max() if not df_full["Year"].empty else selected_year
            trend_agg["Type"] = trend_agg["Year"].apply(lambda x: "Prediction" if x > last_actual_data_year else "Historical")
            trend_agg = trend_agg[(trend_agg['Hotspot_Count'] > 0) | (trend_agg['Year'] >= (min(available_years) if available_years else selected_year))]
            if not trend_agg.empty:
                fig_trend = px.line(
                    trend_agg, x="Year", y="Hotspot_Count", markers=True, color="Type",
                    color_discrete_map={"Historical": "#1F77B4", "Prediction": "#FF7F0E"}, title=f"Hotspot Count Trend for {scope_name}")
                fig_trend.update_layout(showlegend=True, legend_title_text='Data Type', xaxis_title="Year", yaxis_title="Number of Hotspot Counties")
                st.plotly_chart(fig_trend, use_container_width=True)
            else: st.info(f"Not enough historical data for selected counties in {scope_name} to show a trend.")
        else: st.info(f"No counties selected or available in {scope_name} to display trend data.")

    with top_counties_tab: 
        st.subheader(f"🏆 Top Hotspot Counties in {scope_name} ({selected_year})")
        if not data_filtered.empty:
            top_counties_data = data_filtered[data_filtered["Hotspot"] == 1].copy()
            if not top_counties_data.empty:
                if "Hotspot_Prediction_Score" in top_counties_data.columns and top_counties_data["Hotspot_Prediction_Score"].notna().any():
                     top_n = top_counties_data.sort_values(by="Hotspot_Prediction_Score", ascending=False).head(15)
                     y_axis_metric, chart_title = "Hotspot_Prediction_Score", "Top 15 Counties by Hotspot Prediction Score"
                else: 
                    top_n = top_counties_data.head(15); top_n.loc[:, "Hotspot_Indicator"] = 1
                    y_axis_metric, chart_title = "Hotspot_Indicator", "Hotspot Counties (up to 15 shown)"
                fig_bar_counties = px.bar(top_n, x="County", y=y_axis_metric, color="State", title=chart_title, labels={"County": "County Name", y_axis_metric: "Hotspot Score/Indicator", "State": "State"}, hover_data=["State", "County", y_axis_metric])
                fig_bar_counties.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_bar_counties, use_container_width=True)
                st.markdown("---")
                st.subheader(f" States with Most Hotspots ({selected_year})")
                state_hotspot_counts = data_filtered[data_filtered['Hotspot']==1].groupby("State")["County"].nunique().sort_values(ascending=False).reset_index(name="Hotspot_County_Count")
                if not state_hotspot_counts.empty:
                    fig_bar_states = px.bar(state_hotspot_counts.head(10), x="State", y="Hotspot_County_Count", color="State", title=f"Top 10 States by Number of Hotspot Counties in {scope_name}", labels={"State": "State", "Hotspot_County_Count": "Number of Hotspot Counties"})
                    st.plotly_chart(fig_bar_states, use_container_width=True)
                else: st.info(f"No states with hotspots found in {scope_name} for {selected_year}.")
            else: st.info(f"No hotspot counties found in {scope_name} for {selected_year}.")
        else: st.info("No data available for the selected filters to display top counties/states.")

    if pollutant_columns and pollutant_analysis_tab:
        with pollutant_analysis_tab:
            st.subheader(f"🏭 Analysis for Pollutant: {selected_pollutant_sidebar} ({selected_year})")
            if selected_pollutant_sidebar and selected_pollutant_sidebar in data_filtered.columns and data_filtered[selected_pollutant_sidebar].notna().any():
                pollutant_data_for_viz = data_filtered.dropna(subset=[selected_pollutant_sidebar]).copy()
                
                st.markdown(f"**Key Statistics for {selected_pollutant_sidebar} in {scope_name}**")
                stats = pollutant_data_for_viz[selected_pollutant_sidebar].agg(['min', 'max', 'mean', 'median', 'std']).rename(index={'min':'Minimum', 'max':'Maximum', 'mean':'Mean', 'median':'Median', 'std':'Std. Deviation'})
                st.table(stats.apply(lambda x: f"{x:.2f}"))

                st.markdown(f"**Distribution of {selected_pollutant_sidebar} in {scope_name}**")
                fig_hist_pollutant = px.histogram(pollutant_data_for_viz, x=selected_pollutant_sidebar, color="Hotspot", marginal="box", color_discrete_map={0: "#457B9D", 1: "#E63946"}, labels={"Hotspot": "Is Hotspot?"})
                st.plotly_chart(fig_hist_pollutant, use_container_width=True)

                st.markdown(f"**Average {selected_pollutant_sidebar} Levels by Hotspot Status**")
                avg_pollutant_levels = pollutant_data_for_viz.groupby("Hotspot")[selected_pollutant_sidebar].mean().reset_index()
                avg_pollutant_levels["Hotspot"] = avg_pollutant_levels["Hotspot"].map({0: "Not Hotspot", 1: "Hotspot"})
                fig_avg_pollutant = px.bar(avg_pollutant_levels, x="Hotspot", y=selected_pollutant_sidebar, color="Hotspot", color_discrete_map={"Not Hotspot": "#457B9D", "Hotspot": "#E63946"}, title=f"Average {selected_pollutant_sidebar} by Hotspot Status")
                st.plotly_chart(fig_avg_pollutant, use_container_width=True)

                if "Hotspot_Prediction_Score" in pollutant_data_for_viz.columns and pollutant_data_for_viz["Hotspot_Prediction_Score"].notna().any():
                    st.markdown(f"**{selected_pollutant_sidebar} vs. Hotspot Prediction Score**")
                    fig_scatter_pollutant_score = px.scatter(pollutant_data_for_viz, x=selected_pollutant_sidebar, y="Hotspot_Prediction_Score", color="Hotspot", color_discrete_map={0: "#457B9D", 1: "#E63946"}, trendline="ols", trendline_scope="overall", hover_name="County", hover_data=["State"], title=f"{selected_pollutant_sidebar} vs. Hotspot Prediction Score in {scope_name}")
                    st.plotly_chart(fig_scatter_pollutant_score, use_container_width=True)
                
                st.markdown("---")
                st.subheader(f"📅 Monthly Trends for Pollutants in {scope_name} ({selected_year})")
                monthly_groups = st.session_state.get('monthly_pollutant_groups', {})
                if monthly_groups:
                    base_pollutant_for_monthly = st.selectbox("Select Base Pollutant for Monthly Trend", options=list(monthly_groups.keys()), key="base_pollutant_monthly")
                    if base_pollutant_for_monthly and base_pollutant_for_monthly in monthly_groups:
                        monthly_cols_for_selected_base = monthly_groups[base_pollutant_for_monthly]
                        actual_monthly_cols_in_data = [col for col in monthly_cols_for_selected_base if col in data_filtered.columns]
                        if actual_monthly_cols_in_data:
                            monthly_trend_data = data_filtered[actual_monthly_cols_in_data].copy()
                            if not monthly_trend_data.empty:
                                monthly_means = monthly_trend_data.mean().reset_index()
                                monthly_means.columns = ['FullPollutantName', 'AverageValue']
                                month_pattern_plot = re.compile(r"(.+)_(" + "|".join(MONTH_ORDER) + r")$", re.IGNORECASE)
                                monthly_means['Month'] = monthly_means['FullPollutantName'].apply(lambda x: month_pattern_plot.match(x).group(2).capitalize() if month_pattern_plot.match(x) else None)
                                monthly_means.dropna(subset=['Month'], inplace=True)
                                monthly_means['Month'] = pd.Categorical(monthly_means['Month'], categories=MONTH_ORDER, ordered=True)
                                monthly_means.sort_values('Month', inplace=True)
                                if not monthly_means.empty:
                                    fig_monthly_trend = px.line(monthly_means, x='Month', y='AverageValue', markers=True, title=f"Average Monthly Trend for {base_pollutant_for_monthly}")
                                    fig_monthly_trend.update_layout(yaxis_title=f"Average {base_pollutant_for_monthly} Level")
                                    st.plotly_chart(fig_monthly_trend, use_container_width=True)
                                    if not monthly_means.empty:
                                        highest_month = monthly_means.loc[monthly_means['AverageValue'].idxmax()]
                                        st.markdown(f"**Peak Month for {base_pollutant_for_monthly}:** {highest_month['Month']} (Average: {highest_month['AverageValue']:.2f})")
                                else: st.info(f"No valid monthly data to plot for {base_pollutant_for_monthly} in the current selection.")
                            else: st.info(f"No data available for monthly columns of {base_pollutant_for_monthly} in the current selection.")
                        else: st.info(f"None of the expected monthly columns for {base_pollutant_for_monthly} are present in the filtered data.")
                else: st.info("No monthly pollutant groups identified in the data for detailed monthly trend analysis.")
            else: st.info(f"No data or all NaN values for selected pollutant '{selected_pollutant_sidebar}' in the filtered data. Cannot display pollutant analysis.")

    with data_view_tab:
        st.subheader(f"📄 Raw Data for {scope_name} ({selected_year})")
        if not data_filtered.empty:
            st.dataframe(data_filtered, use_container_width=True, height=400)
            csv_data = convert_df_to_csv(data_filtered)
            st.download_button(label="📥 Download Filtered Data as CSV", data=csv_data, file_name=f"hotspot_pollutant_data_{selected_year}_{selected_state.replace(' ','_')}.csv", mime="text/csv")
        else: st.info("No data to display for the current selection.")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>Dashboard created with Streamlit & Plotly Express</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: grey;'>Hotspot data: {HOTSPOT_DATA_PATH}, Pollutant data: {POLLUTANT_DATA_PATH}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    if not (os.path.exists(HOTSPOT_DATA_PATH) and os.path.exists(POLLUTANT_DATA_PATH)):
        st.error("Fatal Error: One or both data files were not found. The application cannot start.")
        st.info(f"Please ensure '{HOTSPOT_DATA_PATH}' and '{POLLUTANT_DATA_PATH}' are in the correct directory or update their paths.")
    else:
        main()