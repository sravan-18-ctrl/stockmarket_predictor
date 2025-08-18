import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to fetch stock data and display it
def fetch_stock_data():
    stock_symbol = entry_symbol.gepipt()
    
    if not stock_symbol:
        messagebox.showerror("Input Error", "Please enter a stock symbol!")
        return
    
    try:
        # Fetch historical data for the stock symbol
        stock_data = yf.download(stock_symbol, period="1mo", interval="1d")
        
        if stock_data.empty:
            messagebox.showerror("Data Error", f"No data found for {stock_symbol}")
            return
        
        # Plot the data (closing price)
        plot_stock_data(stock_data)
    except Exception as e:
        messagebox.showerror("Error", f"Error fetching data: {str(e)}")

# Function to plot the stock data
def plot_stock_data(stock_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(stock_data['Close'], label="Closing Price", color='blue')
    ax.set_title(f"Stock Data for {entry_symbol.get()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    
    # Clear previous plot if any
    for widget in frame_plot.winfo_children():
        widget.destroy()
    
    # Display new plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Create the main application window
root = tk.Tk()
root.title("Stock Market Predictor")
root.geometry("800x600")

# Create input label and entry for stock symbol
label_symbol = tk.Label(root, text="Enter Stock Symbol (e.g., AAPL, TSLA):")
label_symbol.pack(pady=10)

entry_symbol = tk.Entry(root, font=("Arial", 14), width=20)
entry_symbol.pack(pady=5)

# Button to fetch stock data
button_fetch = tk.Button(root, text="Fetch Stock Data", font=("Arial", 14), command=fetch_stock_data)
button_fetch.pack(pady=10)

# Frame for plotting the stock data
frame_plot = tk.Frame(root)
frame_plot.pack(pady=20)

# Start the application
root.mainloop()
