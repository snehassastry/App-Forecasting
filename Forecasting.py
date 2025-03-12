{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7b05bcb1-c80d-4d17-883e-99e8fe8d253e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-12 16:45:34.172 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\rayad\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from statsmodels.tsa.stattools import adfuller, acf, pacf\n",
    "from statsmodels.tsa.arima.model import ARIMA\n",
    "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from pmdarima import auto_arima\n",
    "\n",
    "# Function to plot ACF and PACF\n",
    "def plot_acf_pacf(data, column):\n",
    "    fig, axes = plt.subplots(2, figsize=(10, 6))\n",
    "    axes[0].plot(acf(data[column]))\n",
    "    axes[0].set_title('ACF')\n",
    "    axes[1].plot(pacf(data[column]))\n",
    "    axes[1].set_title('PACF')\n",
    "    return fig\n",
    "\n",
    "# Function to evaluate model performance\n",
    "def evaluate_model(y_test, y_pred):\n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mse)\n",
    "    return rmse\n",
    "\n",
    "# Main Streamlit App\n",
    "def main():\n",
    "    st.title(\"Forecasting Tool\")\n",
    "    \n",
    "    # File Upload\n",
    "    uploaded_file = st.file_uploader(\"Choose a CSV file\", type=['csv'])\n",
    "    \n",
    "    if uploaded_file is not None:\n",
    "        data = pd.read_csv(uploaded_file)\n",
    "        \n",
    "        # Select Column to Forecast\n",
    "        columns = data.columns.tolist()\n",
    "        column_to_forecast = st.selectbox(\"Select column to forecast\", columns)\n",
    "        \n",
    "        # Display Trends\n",
    "        st.write(\"Current Trends:\")\n",
    "        fig, ax = plt.subplots()\n",
    "        ax.plot(data[column_to_forecast])\n",
    "        st.pyplot(fig)\n",
    "        \n",
    "        # Display ACF and PACF\n",
    "        st.write(\"ACF and PACF Plots:\")\n",
    "        acf_pacf_fig = plot_acf_pacf(data, column_to_forecast)\n",
    "        st.pyplot(acf_pacf_fig)\n",
    "        \n",
    "        # Forecasting Models\n",
    "        models = {\n",
    "            \"1. Naive\": \"Simplest model, uses last observed value.\",\n",
    "            \"2. ARIMA\": \"AutoRegressive Integrated Moving Average model.\",\n",
    "            \"3. SARIMA\": \"Seasonal ARIMA model.\",\n",
    "            \"4. AutoARIMA\": \"Automatically selects best ARIMA parameters.\",\n",
    "            \"5. Linear Regression\": \"Basic regression model.\",\n",
    "            \"6. Exponential Smoothing\": \"Simple method for forecasting.\",\n",
    "            \"7. Holt-Winters\": \"Seasonal exponential smoothing.\",\n",
    "            \"8. Prophet\": \"General additive model for forecasting.\",\n",
    "            \"9. LSTM\": \"Long Short-Term Memory neural network.\",\n",
    "            \"10. Vector Autoregression\": \"Model for multiple time series.\"\n",
    "        }\n",
    "        \n",
    "        model_selection = st.selectbox(\"Select a forecasting model\", list(models.keys()))\n",
    "        \n",
    "        # Number of Periods to Forecast\n",
    "        periods_to_forecast = st.number_input(\"Enter number of periods to forecast\", min_value=1)\n",
    "        \n",
    "        if st.button(\"Run Forecast\"):\n",
    "            # Prepare Data\n",
    "            train_data, test_data = train_test_split(data[column_to_forecast], test_size=0.2, random_state=42)\n",
    "            \n",
    "            # Run Selected Model\n",
    "            if model_selection == \"1. Naive\":\n",
    "                forecast = [train_data.iloc[-1]] * periods_to_forecast\n",
    "            elif model_selection == \"2. ARIMA\":\n",
    "                model = ARIMA(train_data, order=(1,1,1))\n",
    "                model_fit = model.fit()\n",
    "                forecast = model_fit.forecast(steps=periods_to_forecast)\n",
    "            elif model_selection == \"3. SARIMA\":\n",
    "                model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,12))\n",
    "                model_fit = model.fit()\n",
    "                forecast = model_fit.forecast(steps=periods_to_forecast)\n",
    "            elif model_selection == \"4. AutoARIMA\":\n",
    "                model = auto_arima(train_data, start_p=1, start_d=1, start_q=1, max_p=5, max_d=5, max_q=5)\n",
    "                forecast = model.predict(n_periods=periods_to_forecast)\n",
    "            else:\n",
    "                st.error(\"Model not implemented yet.\")\n",
    "                return\n",
    "            \n",
    "            # Display Forecast\n",
    "            forecast_df = pd.DataFrame(forecast, columns=['Forecast'])\n",
    "            st.write(\"Forecast:\")\n",
    "            st.write(forecast_df)\n",
    "            \n",
    "            # Download Forecast\n",
    "            @st.cache\n",
    "            def convert_df(df):\n",
    "                return df.to_csv(index=False).encode('utf-8')\n",
    "            \n",
    "            csv = convert_df(forecast_df)\n",
    "            st.download_button(\"Download Forecast\", csv, \"forecast.csv\", \"text/csv\", key='download-csv')\n",
    "            \n",
    "            # Evaluate Model Performance\n",
    "            if len(test_data) >= periods_to_forecast:\n",
    "                y_pred = forecast[:len(test_data)]\n",
    "                rmse = evaluate_model(test_data, y_pred)\n",
    "                st.write(f\"RMSE: {rmse}\")\n",
    "            \n",
    "            # Optional: Run All Models and Rank by Accuracy\n",
    "            # This part is complex and requires implementing all models and calculating their RMSE.\n",
    "            # For simplicity, it's not fully implemented here but can be added similarly to the above logic.\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73ba0325-0d64-4826-84f7-d06012d2164f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ceab4b4c-b5f9-4948-9213-d90878841aa1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7324e9b-eb48-42ce-978a-dd028ab401c1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "879c02cd-e7cf-40e3-804f-c0354767f5c3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53fdf199-d5a2-4bbc-b702-cdc63064017a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43b28450-4ef0-44e3-86aa-313873d5d228",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d171e74b-680a-4f64-8f4f-6cd33f4aac4f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f9ecc38-1e13-44ec-872c-4a3d76dc38e0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1d83a37-a79b-4cf7-97c7-561c2ef422e7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9f54e7f-c4c5-480c-9969-df2c80091af2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95a81938-a519-4c44-9571-22aef3705f9c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95b21adf-d324-4f4c-af59-1e68dd85f989",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1129c95b-daf9-471b-b0a8-f10ebec55bd2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb7b53f1-ad7c-4465-b5ad-0c1adefc9fc6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56091c7f-fc50-4ad3-a078-8e8f19b8917c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06cf34a6-59ab-4cb4-8f6a-ab1093784c44",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb35453f-d53f-4519-a383-761f5135a394",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3be2af6-c4e2-46be-8440-002081a2eb10",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1663329-5eae-4809-a4f2-f7cce99e71cd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8302615-10bd-4f19-ba23-a70c00f750ff",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73b18260-c5d8-4554-9fac-cf65ea08d8a5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "489b2a4c-ff68-4df2-bebc-c5c355a83efa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
