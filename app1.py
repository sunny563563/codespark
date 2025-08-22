import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
data = pd.read_csv('Rural_Development.csv')

# Fix column name typos if needed
data.rename(columns={
    'Population (Milions)': 'Population (Millions)',
    'Hosipitals Dvelopment (%)': 'Hospitals Development (%)',
    'Education Rate (%)': 'Education Development (%)'  # adjust as you want
}, inplace=True)

st.title("Rural Development Dashboard with Multiple Charts")

# Sidebar filters
years = sorted(data['Year'].unique())
states = sorted(data['State'].unique())

selected_year = st.sidebar.selectbox("Select Year", years)
selected_state = st.sidebar.selectbox("Select State", states)

# Filter data
filtered_data = data[(data['Year'] == selected_year) & (data['State'] == selected_state)]

if filtered_data.empty:
    st.warning("No data available for the selected year and state.")
else:
    st.header(f"Data for {selected_state} in {selected_year}")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Population (Millions)", filtered_data['Population (Millions)'].values[0])
    col2.metric("Unemployment Rate (%)", filtered_data['Unemployment Rate (%)'].values[0])
    col3.metric("Per Capita Income (INR)", f"{filtered_data['Per Capita Income (INR)'].values[0]:,.2f}")

    # Prepare data for charts
    dev_metrics = {
        'Roads Development (%)': filtered_data['Roads Development (%)'].values[0],
        'Hospitals Development (%)': filtered_data['Hospitals Development (%)'].values[0],
        'Housing Development (%)': filtered_data['Housing Development (%)'].values[0],
        'Education Development (%)': filtered_data['Education Development (%)'].values[0]
    }

    # Convert to DataFrame for Plotly
    dev_df = pd.DataFrame({
        'Metric': list(dev_metrics.keys()),
        'Value': list(dev_metrics.values())
    })

    # 1. Column Chart (Bar chart vertical)
    st.subheader("Column Chart - Development Metrics")
    fig_col = px.bar(dev_df, x='Metric', y='Value', color='Metric', labels={'Value':'Percentage'}, range_y=[0, 100])
    st.plotly_chart(fig_col, use_container_width=True)

    # 2. Bar Chart (Bar chart horizontal)
    st.subheader("Bar Chart - Development Metrics")
    fig_bar = px.bar(dev_df, y='Metric', x='Value', color='Metric', orientation='h', labels={'Value':'Percentage'}, range_x=[0, 100])
    st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Pie Chart - Distribution of development efforts
    st.subheader("Pie Chart - Development Metrics Distribution")
    fig_pie = px.pie(dev_df, names='Metric', values='Value', color='Metric')
    st.plotly_chart(fig_pie, use_container_width=True)

    # 4. Line Chart - Example: Trends over years for selected state
    st.subheader(f"Line Chart - Trends over Years for {selected_state}")

    # Filter data for the state across all years
    state_data = data[data['State'] == selected_state].sort_values('Year')

    # Prepare line chart data for one metric or multiple
    fig_line = px.line(state_data, x='Year', y=['Roads Development (%)', 'Hospitals Development (%)', 'Housing Development (%)', 'Education Development (%)'],
                       labels={'value': 'Percentage', 'variable': 'Metric'},
                       title=f"Development Trends in {selected_state} Over Years")
    st.plotly_chart(fig_line, use_container_width=True)
