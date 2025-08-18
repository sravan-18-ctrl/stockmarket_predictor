import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Read the Apple stock data
data = pd.read_csv('Apple.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')  # Ensure the 'Date' is in datetime format

# Check the shape and sample of the data
print(data.shape)
print(data.sample(7))
data.info()

# Plot initial data
plt.plot(data['Date'], data['Close'], c="r", label="Close", marker="+")
plt.plot(data['Date'], data['Open'], c="g", label="Open", marker="^")
plt.legend()
plt.tight_layout()

# Prepare the data for training
close_data = data[['Close']]
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Split the data into training and testing data
train_data = scaled_data[:training, :]
x_train = []
y_train = []

# Prepare training data features and labels
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=10, batch_size=32)

# Prepare test data
test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict the stock prices using the trained model
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
rmse = np.sqrt(mse)

print("MSE", mse)
print("RMSE", rmse)

# Combine test data with predictions
test = data[training:]
test['Predictions'] = predictions

# Plot the actual vs predicted stock prices
# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 8))

# Plot the training data (just use 'Date' and 'Close' from 'data' for this)
plt.plot(data['Date'][:training], data['Close'][:training], label="Train")

# Plot the actual test data
plt.plot(test['Date'], test['Close'], label="Test")

# Plot the predicted stock prices
plt.plot(test['Date'], test['Predictions'], label="Predictions")

plt.title('Apple Stock Close Price Prediction')
plt.xlabel('Date')
plt.ylabel("Close Price (USD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# Tkinter-based GUI for stock prediction
def predict_stock(stock_symbol, start_date, end_date):
    # In a real-world application, this function would use an ML model to predict stock prices
    # For demonstration, we'll return some dummy data
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    prices = np.random.randint(100, 200, size=len(dates))
    return pd.DataFrame({'Date': dates, 'Predicted Price': prices})

# Function to handle the prediction request
def on_predict_button_click():
    stock_symbol = stock_symbol_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if not stock_symbol or not start_date or not end_date:
        messagebox.showerror("Input Error", "Please fill in all fields.")
        return

    try:
        # Here, you can call the function to predict stock prices
        predicted_data = predict_stock(stock_symbol, start_date, end_date)
        
        # Display the result in a new window
        result_window = tk.Toplevel(root)
        result_window.title(f"Prediction for {stock_symbol}")
        
        # Display the predicted data
        result_text = tk.Text(result_window, width=50, height=10)
        result_text.pack(padx=10, pady=10)
        result_text.insert(tk.END, predicted_data.to_string(index=False))
        
        # Plot the prediction graph
        plt.plot(predicted_data['Date'], predicted_data['Predicted Price'])
        plt.title(f"Predicted Stock Prices for {stock_symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main Tkinter window
root = tk.Tk()
root.title("Stock Price Predictor")

# Create and place the widgets
tk.Label(root, text="Enter Stock Symbol (Ticker):").grid(row=0, column=0, padx=10, pady=5)
stock_symbol_entry = tk.Entry(root, width=20)
stock_symbol_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5)
start_date_entry = tk.Entry(root, width=20)
start_date_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=5)
end_date_entry = tk.Entry(root, width=20)
end_date_entry.grid(row=2, column=1, padx=10, pady=5)

# Predict button
predict_button = tk.Button(root, text="Predict Stock Price", command=on_predict_button_click)
predict_button.grid(row=3, column=0, columnspan=2, pady=10)

# Run the Tkinter main loop
root.mainloop()
