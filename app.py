import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ---- Streamlit Page Config ----
st.set_page_config(page_title="ML Forecast Dashboard", layout="centered")

# ---- Sidebar Navigation ----
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Inflation", "Predict Unemployment", "History", "About"])

# ---- Initialize Session State ----
if 'history' not in st.session_state:
    st.session_state.history = []

# ---- Load Inflation Data ----
@st.cache_data
def load_inflation_data():
    df = pd.read_csv('global_inflation_data.csv')
    year_cols = [str(year) for year in range(1980, 2025)]
    df_long = df.melt(id_vars=['country_name', 'indicator_name'],
                      value_vars=year_cols,
                      var_name='year',
                      value_name='inflation_rate')
    df_long['year'] = df_long['year'].astype(int)
    df_long = df_long.dropna(subset=['inflation_rate'])
    return df_long

# ---- Load Unemployment Data ----
@st.cache_data
def load_unemployment_data():
    df = pd.read_csv("Cleaned_Unemployment_in_India.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if 'date' not in df.columns:
        st.error("❌ 'date' column not found. Please check your CSV file.")
        st.stop()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

# ---- Helper Functions for Unemployment Prediction ----
def prepare_state_data(df, state):
    state_df = df[df['region'].str.lower() == state.lower()].copy()
    if state_df.empty:
        return None
    state_df = state_df.sort_values(by='date')
    state_df['unemployment_rate_lag1'] = state_df['estimated_unemployment_rate_(%)'].shift(1)
    return state_df.dropna()

def train_model_for_state(df):
    X = df[['year', 'month', 'unemployment_rate_lag1']]
    y = df['estimated_unemployment_rate_(%)']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_unemployment_rate(df, state, year):
    state_df = prepare_state_data(df, state)
    if state_df is None:
        return f"No data available for '{state}'", None, None
    model = train_model_for_state(state_df)
    past_data = state_df[state_df['year'] < year]
    if past_data.empty:
        return f"Not enough historical data for {state} to predict {year}", None, None
    last_known = past_data.sort_values('date').iloc[-1]
    last_lag = last_known['estimated_unemployment_rate_(%)']
    predictions = []
    months = list(range(1, 13))
    for month in months:
        X_pred = pd.DataFrame({
            'year': [year],
            'month': [month],
            'unemployment_rate_lag1': [last_lag]
        })
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        last_lag = pred
    actual_data = state_df[state_df['year'] == year - 1].copy().sort_values(by='month')
    return f"Predicted average unemployment rate for {state.title()} in {year}: {np.mean(predictions):.2f}%", actual_data, predictions

def plot_unemployment_predictions(state, year, actual_data, predictions):
    months = list(range(1, 13))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig, ax = plt.subplots(figsize=(10, 5))
    if actual_data is not None and not actual_data.empty:
        ax.bar(months, actual_data['estimated_unemployment_rate_(%)'],
               width=0.4, label=f'Actual {year - 1}', align='center')
    ax.bar([m + 0.4 for m in months], predictions,
           width=0.4, label=f'Predicted {year}', color='orange', align='center')
    ax.set_xticks([m + 0.2 for m in months])
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Month')
    ax.set_ylabel('Unemployment Rate (%)')
    ax.set_title(f"{state.title()} - Unemployment Prediction")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

# ---- Page: Home ----
if page == "Home":
    st.title("🏠 Welcome to the Economic Forecast Dashboard")
    st.markdown("""
    Use the sidebar to navigate between:

    - 📊 Predict **Inflation Rates** globally
    - 📉 Forecast **Unemployment Rates** in Indian states
    - 🕓 View past **Prediction History**
    - ℹ️ Learn more **About** this app
    """)

# ---- Page: Predict Inflation ----
elif page == "Predict Inflation":
    st.subheader("📊 Inflation Rate Prediction")
    df_inflation = load_inflation_data()
    countries = sorted(df_inflation['country_name'].unique())
    selected_country = st.selectbox("Select a Country", countries)
    future_year = st.number_input("Enter Year (greater than 2024)", min_value=2025, step=1)
    df_country = df_inflation[df_inflation['country_name'] == selected_country]
    if not df_country.empty:
        X = df_country[['year']]
        y = df_country['inflation_rate']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        predicted_value = model.predict(pd.DataFrame({'year': [future_year]}))[0]
        st.success(f"📌 Predicted Inflation Rate for {selected_country} in {future_year}: **{predicted_value:.2f}%**")

        # Save to history
        st.session_state.history.append({
            'type': 'Inflation',
            'location': selected_country,
            'year': future_year,
            'prediction': round(predicted_value, 2)
        })

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_country['year'], df_country['inflation_rate'], marker='o', label='Historical Data')
        ax.plot(future_year, predicted_value, marker='x', color='red', label='Predicted')
        ax.set_title(f"{selected_country} - Inflation Rate Forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Inflation Rate (%)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ---- Page: Predict Unemployment ----
elif page == "Predict Unemployment":
    st.subheader("📉 Unemployment Rate Prediction")
    df_unemp = load_unemployment_data()
    states = sorted(df_unemp['region'].unique())
    selected_state = st.selectbox("Select a State/Region", states)
    future_year_unemp = st.number_input("Enter Year (e.g., 2025)", min_value=2025, step=1, key="unemp_year")
    result_text, actual_data, predictions = predict_unemployment_rate(df_unemp, selected_state, future_year_unemp)
    st.info(result_text)
    if predictions:
        avg_prediction = round(np.mean(predictions), 2)
        # Save to history
        st.session_state.history.append({
            'type': 'Unemployment',
            'location': selected_state,
            'year': future_year_unemp,
            'prediction': avg_prediction
        })
        plot_unemployment_predictions(selected_state, future_year_unemp, actual_data, predictions)

# ---- Page: History ----
elif page == "History":
    st.subheader("🕓 Prediction History")
    if not st.session_state.history:
        st.info("No predictions yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

# ---- Page: About ----
elif page == "About":
    st.subheader("ℹ️ About This App")
    st.markdown("""
    **Economic Forecast Dashboard** is a machine learning-powered tool that allows users to forecast:

    - 📊 **Inflation Rates** for countries using historical data from 1980–2024.
    - 📉 **Unemployment Rates** for Indian states/regions using recent monthly data.

    This tool leverages models such as:
    - `RandomForestRegressor` for inflation rate forecasting.
    - `LinearRegression` with lag features for unemployment rate prediction.

    All predictions are illustrative and should not be used for investment or policy decisions.
    """)
