import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

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

# Function to calculate future stock price using a simple model (exponential growth)
def predict_future_price(df, days_ahead=30):
    if df is None or df.empty:
        return None
    last_price = df['Close'].iloc[-1]
    # Let's assume a simple daily growth rate based on the average daily return
    daily_return = df['Close'].pct_change().mean()
    future_price = last_price * (1 + daily_return) ** days_ahead
    return future_price

# Main Streamlit application
def main():
    st.title("Stock Price Prediction App")
    
    # User input for stock ticker
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):", value='TATAMOTORS.NS')
    
    # User input for time period
    time_period = st.number_input("Enter time period to predict (in days):", min_value=1, max_value=365, value=30)
    
    # Fetch the stock data
    if ticker:
        st.write(f"Fetching data for {ticker}...")
        stock_data = fetch_stock_data(ticker)
        
        if stock_data is not None:
            # Plot stock's closing price over time
            fig, ax = plt.subplots()
            ax.plot(stock_data.index, stock_data['Close'], label="Close Price")
            ax.set_title(f"{ticker} Stock Price Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)
            
            # Predict future stock price
            future_price = predict_future_price(stock_data, time_period)
            if future_price is not None:
                st.write(f"Predicted price for {ticker} in {time_period} days: ${future_price.iloc[0]:.2f}")

            else:
                st.error("Could not calculate future price. Please check the data.")
        else:
            st.error(f"Failed to retrieve data for {ticker}. Please check the ticker symbol or try again later.")
            
if __name__ == "__main__":
    main()
