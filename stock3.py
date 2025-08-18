import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date='2020-01-01'):
    try:
        stock_data = yf.download(ticker, start=start_date)
        if stock_data.empty:
            raise ValueError("No data found for this ticker symbol.")
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Function to prepare data for LSTM model
def prepare_lstm_data(df):
    close_data = df[['Close']]
    dataset = close_data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    training_data_len = int(np.ceil(len(dataset) * 0.95))
    train_data = scaled_data[:training_data_len]
    
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return scaler, x_train, y_train, training_data_len, scaled_data

# Build the LSTM model
def build_lstm_model(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.LSTM(units=64))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future price using exponential growth model
def predict_future_price(df, days_ahead=30):
    if df is None or df.empty:
        return None
    
    # Access last closing price
    last_price = df['Close'].iloc[-1]
    
    if isinstance(last_price, pd.Series):  # Ensure it's a scalar value
        last_price = last_price.item()
    
    # Calculate the daily return
    daily_return = df['Close'].pct_change().mean()
    
    if daily_return is None:  # If there is no return data
        return None
    
    # Apply exponential growth model for prediction
    future_price = last_price * (1 + daily_return) ** days_ahead
    return future_price

# Function to predict future high and low prices based on exponential growth model
def predict_future_high_low(df, days_ahead=30, high_deviation=0.03, low_deviation=0.03):
    if df is None or df.empty:
        return None, None
    
    # Access last closing price
    last_price = df['Close'].iloc[-1]
    
    if isinstance(last_price, pd.Series):  # Ensure it's a scalar value
        last_price = last_price.item()
    
    daily_return = df['Close'].pct_change().mean()
    
    if daily_return is None:  # In case there are issues with daily return calculation
        return None, None
    
    # Predicted future price
    future_price = last_price * (1 + daily_return) ** days_ahead
    
    # Predicted high and low prices based on deviation
    predicted_high = future_price * (1 + high_deviation)  # Assume high is 3% higher
    predicted_low = future_price * (1 - low_deviation)   # Assume low is 3% lower
    
    return predicted_high, predicted_low

# Main Streamlit application
def main():
    st.title("📈 Stock Price Prediction App")
    
    # User input for stock ticker
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):", value='AAPL')
    
    # User input for time period
    time_period = st.number_input("Enter time period to predict (in days):", min_value=1, max_value=365, value=30)
    
    # Fetch the stock data
    if ticker:
        st.write(f"Fetching data for {ticker}...")
        stock_data = fetch_stock_data(ticker)
        
        if stock_data is not None:
            # Display the current price
            current_price = stock_data['Close'].iloc[-1]  # Last closing price
            if isinstance(current_price, pd.Series):  # Ensure it's scalar
                current_price = current_price.item()
            st.write(f"Current price for {ticker}: ${current_price:.2f}")
            
            # Plot stock's closing price over time
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stock_data.index, stock_data['Close'], label="Close Price")
            ax.set_title(f"{ticker} Stock Price Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)
            
            # Prepare the data for LSTM model
            scaler, x_train, y_train, training_data_len, scaled_data = prepare_lstm_data(stock_data)
            
            # Build and train the LSTM model
            model = build_lstm_model((x_train.shape[1], 1))
            model.fit(x_train, y_train, epochs=10, batch_size=32)
            
            # Prepare test data
            test_data = scaled_data[training_data_len - 60:]
            x_test = []
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])
            
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Make predictions with LSTM
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
            # Plotting predictions with corrected date handling
            test = stock_data.iloc[training_data_len:].copy()  # Use iloc to avoid index-related issues
            test['Predictions'] = predictions.flatten()  # Flatten predictions  
                        
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stock_data.index[:training_data_len], stock_data['Close'][:training_data_len], label="Train")
            ax.plot(test.index, test['Close'], label="Test")
            ax.plot(test.index, test['Predictions'], label="Predictions")
            ax.set_title(f"{ticker} Stock Price Predictions (LSTM)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)
            
            # Simple Future Price Prediction using Exponential Growth
            future_price = predict_future_price(stock_data, time_period)
            
            # Check the type of `future_price`
            print(f"Type of future_price: {type(future_price)}")
            if future_price is not None:
                # Ensure future_price is a scalar value (float)
                if isinstance(future_price, pd.Series):
                    future_price = future_price.item()
                
                st.write(f"Predicted price for {ticker} in {time_period} days (Exponential Growth): ${future_price:.2f}")
                
                # Predict high and low for the future day
                predicted_high, predicted_low = predict_future_high_low(stock_data, time_period)
                
                # Check the types of predicted high and low
                print(f"Predicted High: {predicted_high}, Type: {type(predicted_high)}")
                print(f"Predicted Low: {predicted_low}, Type: {type(predicted_low)}")
                
                if predicted_high is not None and predicted_low is not None:
                    if isinstance(predicted_high, pd.Series):
                        predicted_high = predicted_high.item()
                    if isinstance(predicted_low, pd.Series):
                        predicted_low = predicted_low.item()
                        
                    st.write(f"Predicted High Price for {ticker} in {time_period} days: ${predicted_high:.2f}")
                    st.write(f"Predicted Low Price for {ticker} in {time_period} days: ${predicted_low:.2f}")
                else:
                    st.error("Could not calculate high and low prices.")
                
            else:
                st.error("Could not calculate future price.")
                
if __name__ == "__main__":
    main()
