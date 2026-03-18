# Stock Price Prediction — LSTM Neural Network

A machine learning project that uses Long Short-Term Memory (LSTM) neural networks to forecast stock prices from historical market data — built end-to-end from data preprocessing through model training and evaluation.

---

## Overview

Traditional time-series models struggle with long-range dependencies in stock data. This project uses LSTM networks — a type of recurrent neural network designed for sequential data — to capture patterns across long historical windows and generate short-term price forecasts.

---

## Results

| Metric | Value |
|--------|-------|
| Model | LSTM (2 layers) |
| Dataset | Historical stock data (add source, e.g. Yahoo Finance) |
| Training period | Add date range |
| Evaluation metric | RMSE / MAE — _add your actual number here_ |
| Normalisation | MinMaxScaler (0–1 range) |

> 💡 **Tip:** Replace the placeholder metrics above with your actual results — even approximate numbers (e.g. "RMSE of 3.2 on 30-day test window") make this entry significantly stronger.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Deep Learning | TensorFlow, Keras |
| ML Utilities | Scikit-learn |
| Data Processing | NumPy, pandas |
| Visualisation | Matplotlib |
| Environment | Jupyter Notebook / VS Code |

---

## Model Architecture

```
Input Layer  →  [sequence_length, features]
     ↓
LSTM Layer 1 →  units=50, return_sequences=True
     ↓
Dropout      →  0.2
     ↓
LSTM Layer 2 →  units=50, return_sequences=False
     ↓
Dropout      →  0.2
     ↓
Dense Layer  →  units=1 (predicted price)
```

---

## Getting Started

### Prerequisites

```bash
python >= 3.9
pip
```

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook stock_prediction.ipynb
```

---

## Project Structure

```
stock-price-prediction/
├── data/
│   └── stock_data.csv          # Historical price data
├── models/
│   └── lstm_model.h5           # Saved trained model
├── notebooks/
│   └── stock_prediction.ipynb  # Main notebook
├── src/
│   ├── preprocess.py           # Data loading & MinMaxScaler
│   ├── model.py                # LSTM architecture
│   └── evaluate.py             # Metrics & plots
├── requirements.txt
└── README.md
```

---

## Pipeline Walkthrough

### 1. Data Preprocessing
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
```
MinMaxScaler normalises all values to [0, 1], stabilising gradient descent and speeding up convergence.

### 2. Sequence Construction
```python
# Build sliding windows of length `sequence_length`
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])
```

### 3. Model Training
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---

## Sample Output

> _Add a chart image here (actual vs predicted prices plot) — this is the most impactful thing you can add_

```
Predicted vs Actual — last 30 days
Actual:    [142.3, 144.1, 143.8, 146.2, ...]
Predicted: [141.9, 143.7, 144.2, 145.8, ...]
```

---

## Key Learnings

- How LSTM networks retain context across long sequences, outperforming vanilla RNNs on time-series
- Why normalisation (MinMaxScaler) is critical before training — unnormalised stock prices cause gradient instability
- Trade-offs between sequence length, model depth, and overfitting
- End-to-end ML project structure: ingestion → preprocessing → training → evaluation → export

---

## Future Improvements

- [ ] Add technical indicators (RSI, MACD, Bollinger Bands) as additional features
- [ ] Build a Flask API to serve predictions in real time
- [ ] Deploy on AWS EC2 with a simple React dashboard frontend
- [ ] Experiment with Transformer-based models (e.g. Temporal Fusion Transformer)

---

## Author

**B. Sravan Reddy**
[LinkedIn](https://www.linkedin.com/in/bsravanreddy/) · [GitHub](https://github.com/YOUR_USERNAME)

---

_B.Tech Computer Science · MLR Institute of Technology · 2025_
