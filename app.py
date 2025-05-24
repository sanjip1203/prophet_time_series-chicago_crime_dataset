import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from prophet import Prophet

st.title("📈 Crime Rate Forecasting App (Prophet)")

# Load the model
@st.cache_resource
def load_model():
    import os

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "prophet_model.pkl")
    return joblib.load(MODEL_PATH)


model = load_model()

# Forecast period input
periods_input = st.number_input("Enter number of days to forecast", min_value=1, max_value=365*10, value=365)

# Make future dataframe
future = model.make_future_dataframe(periods=periods_input)

# Predict
forecast = model.predict(future)

# Show raw forecast data
if st.checkbox("Show forecast data"):
    st.subheader("Forecast Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot forecast
st.subheader("Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot forecast components
st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.markdown("---")
st.markdown("✅ Model: Prophet | 📦 Forecast saved as 'prophet_model.pkl'")
