import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date

# Load trained model
model_path = 'Stock Predictions Model.keras'
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Custom CSS for modern UI
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4F8BF9;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1F2937;
    }
    /* Input fields */
    .stTextInput>div>div>input, .stDateInput>div>div>input {
        background-color: #1F2937;
        color: #FAFAFA;
        border: 1px solid #4F8BF9;
        border-radius: 8px;
        padding: 10px;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #3B6BB5;
    }
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #4F8BF9;
    }
    /* Plot background */
    .stPlot {
        background-color: #1F2937;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 10px;
        color: #6B7280;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title('📈 Stock Market Predictor')
st.markdown("""
    Welcome to the **Stock Market Predictor**! This app uses a trained LSTM model to predict stock prices.
    Enter a stock symbol, select a date range, and explore the predictions.
    """)

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Settings")
    stock = st.text_input('Enter Stock Symbol (e.g., GOOG, AAPL, MSFT)', 'GOOG')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Start Date", date(2012, 1, 1))
    with col2:
        end_date = st.date_input("📅 End Date", date(2022, 12, 31))
    num_days = st.slider("🔮 Predict Next (Days)", min_value=10, max_value=100, value=50, step=10)

# Ensure start date is before end date
if start_date >= end_date:
    st.error("⚠️ Start date must be earlier than the end date. Please select a valid range.")
    st.stop()

# Convert dates to string format
start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

# Download stock data
data = yf.download(stock, start, end, progress=False)

# Check if data is empty
if data.empty:
    st.error("⚠️ No data found for the given stock symbol and date range. Try a different selection.")
    st.stop()

# Display stock data
st.subheader('📊 Stock Data')
st.dataframe(data.style.background_gradient(cmap='Blues'))

# Train-test split
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# Check if train/test data is empty
if data_train.empty or data_test.empty:
    st.error("⚠️ Insufficient data to process. Try a different stock or date range.")
    st.stop()



# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scaled = scaler.fit_transform(data_train)

# Prepare test data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

# Check if test data is valid before scaling
if not data_test.empty:
    data_test_scaled = scaler.transform(data_test)
else:
    st.error("⚠️ Test data is empty. Cannot apply MinMaxScaler.")
    st.stop()

# Moving Averages
st.subheader('📈 Moving Averages')
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(data['Close'], 'g', label='Stock Price')
ax.plot(ma_50_days, 'r', label='50-day MA')
ax.plot(ma_100_days, 'b', label='100-day MA')
ax.plot(ma_200_days, 'y', label='200-day MA')
ax.set_facecolor('#1F2937')
ax.set_title('Stock Price vs Moving Averages', color='white')
ax.legend()
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
st.pyplot(fig)


# Prepare data for model prediction
x_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
predictions = model.predict(x_test)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot results
st.subheader('📉 Original Price vs Predicted Price')
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.plot(y_test, 'g', label='Original Price')
ax2.plot(predictions, 'r', label='Predicted Price')
ax2.set_facecolor('#1F2937')
ax2.set_title('Original vs Predicted Price', color='white')
ax2.legend()
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
st.pyplot(fig2)

# --------- Next N Days Prediction ---------
st.subheader(f"🔮 Prediction for Next {num_days} Days")

# Extract last 100 days of data
last_100_days = data_test_scaled[-100:]
last_100_days = last_100_days.reshape(1, 100, 1)  # Reshape to match model input

future_predictions = []

# Predict for user-selected days
for _ in range(num_days):
    future_pred = model.predict(last_100_days)  # Shape: (1, 1)
    future_predictions.append(future_pred[0, 0])  # Extract single value
    # Update last_100_days with the new prediction
    future_pred_reshaped = future_pred.reshape(1, 1, 1)  # Match input shape
    last_100_days = np.append(last_100_days[:, 1:, :], future_pred_reshaped, axis=1)

# Convert predictions back to original scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Plot predicted prices for the selected days
fig3, ax3 = plt.subplots(figsize=(10,6))
ax3.plot(future_predictions, 'r', label='Predicted Price')
ax3.set_facecolor('#1F2937')
ax3.set_title(f'Predicted Price for Next {num_days} Days', color='white')
ax3.set_xlabel('Days')
ax3.set_ylabel('Price')
ax3.legend()
ax3.tick_params(colors='white')
ax3.spines['bottom'].set_color('white')
ax3.spines['left'].set_color('white')
st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown('<p class="footer">Disclaimer: This app is for educational purposes only. Predictions are based on historical data and may not be accurate.</p>', unsafe_allow_html=True)