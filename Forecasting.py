import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

# Function to plot ACF and PACF
def plot_acf_pacf(data, column):
    fig, axes = plt.subplots(2, figsize=(10, 6))
    axes[0].plot(acf(data[column]))
    axes[0].set_title('ACF')
    axes[1].plot(pacf(data[column]))
    axes[1].set_title('PACF')
    return fig

# Function to evaluate model performance
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# Main Streamlit App
def main():
    st.title("Forecasting Tool")
    
    # File Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Select Column to Forecast
        columns = data.columns.tolist()
        column_to_forecast = st.selectbox("Select column to forecast", columns)
        
        # Display Trends
        st.write("Current Trends:")
        fig, ax = plt.subplots()
        ax.plot(data[column_to_forecast])
        st.pyplot(fig)
        
        # Display ACF and PACF
        st.write("ACF and PACF Plots:")
        acf_pacf_fig = plot_acf_pacf(data, column_to_forecast)
        st.pyplot(acf_pacf_fig)
        
        # Forecasting Models
        models = {
            "1. Naive": "Simplest model, uses last observed value.",
            "2. ARIMA": "AutoRegressive Integrated Moving Average model.",
            "3. SARIMA": "Seasonal ARIMA model.",
            "4. AutoARIMA": "Automatically selects best ARIMA parameters.",
            "5. Linear Regression": "Basic regression model.",
            "6. Exponential Smoothing": "Simple method for forecasting.",
            "7. Holt-Winters": "Seasonal exponential smoothing.",
            "8. Prophet": "General additive model for forecasting.",
            "9. LSTM": "Long Short-Term Memory neural network.",
            "10. Vector Autoregression": "Model for multiple time series."
        }
        
        model_selection = st.selectbox("Select a forecasting model", list(models.keys()))
        
        # Number of Periods to Forecast
        periods_to_forecast = st.number_input("Enter number of periods to forecast", min_value=1)
        
        if st.button("Run Forecast"):
            # Prepare Data
            train_data, test_data = train_test_split(data[column_to_forecast], test_size=0.2, random_state=42)
            
            # Run Selected Model
            if model_selection == "1. Naive":
                forecast = [train_data.iloc[-1]] * periods_to_forecast
            elif model_selection == "2. ARIMA":
                model = ARIMA(train_data, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=periods_to_forecast)
            elif model_selection == "3. SARIMA":
                model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=periods_to_forecast)
            elif model_selection == "4. AutoARIMA":
                model = auto_arima(train_data, start_p=1, start_d=1, start_q=1, max_p=5, max_d=5, max_q=5)
                forecast = model.predict(n_periods=periods_to_forecast)
            else:
                st.error("Model not implemented yet.")
                return
            
            # Display Forecast
            forecast_df = pd.DataFrame(forecast, columns=['Forecast'])
            st.write("Forecast:")
            st.write(forecast_df)
            
            # Download Forecast
            @st.cache_data  # Updated caching decorator for Streamlit v1.x+
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(forecast_df)
            st.download_button("Download Forecast", csv, "forecast.csv", "text/csv", key='download-csv')
            
            # Evaluate Model Performance
            if len(test_data) >= periods_to_forecast:
                y_pred = forecast[:len(test_data)]
                rmse = evaluate_model(test_data.values[:len(y_pred)], y_pred)
                st.write(f"RMSE: {rmse}")
            
if __name__ == "__main__":
    main()
