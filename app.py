import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date


# Set page configuration
st.set_page_config(page_title="Stock Price",
                   layout="wide",
                   page_icon="📈")



# Load the model
model = load_model('C:/Users/saivi/OneDrive/Desktop/stock/Stock Predictions Model.keras')

# Create a Streamlit app
st.title("Stock Price Predictor")
st.markdown("Welcome to our Stock Price Predictor! Please enter a stock symbol to get started.")

# Get the stock symbol from the user
stock = st.text_input("Enter Stock Symbol", value="GOOG", max_chars=5)

# Define the start and end dates
start = "2015-01-01"
end = date.today().strftime("%Y-%m-%d")

# Download the stock data
data = yf.download(stock, start, end)

# Display the stock data
st.subheader("Stock Data")
st.write(data)

# Split the data into training and testing sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Create a dataframe with the last 100 days of training data and the testing data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot the price vs MA50
st.subheader("Price vs MA50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)

# Plot the price vs MA50 vs MA100
st.subheader("Price vs MA50 vs MA100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Plot the price vs MA100 vs MA200
st.subheader("Price vs MA100 vs MA200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare the data for prediction
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)

# Scale the predictions
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Plot the original price vs predicted price
st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
