# Finance AI — Stock Price Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?style=flat-square)
![Backtrader](https://img.shields.io/badge/Backtrader-Backtesting-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

## Overview

An end-to-end machine learning system that predicts directional stock price movement using a deep **LSTM (Long Short-Term Memory)** neural network trained on engineered financial time-series features. The project covers the full data science pipeline: data ingestion via SQL-queried API, feature engineering, model training, automated backtesting, and a React-based portfolio performance dashboard.

**Result:** Achieved above-baseline directional prediction accuracy, validated by ROC curve analysis and Sharpe ratio metrics against a buy-and-hold benchmark.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10, SQL |
| ML Framework | TensorFlow / Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Data Source | yfinance API (SQL-queried via PostgreSQL) |
| Backtesting | Backtrader |
| Visualization | Matplotlib, React (dashboard frontend) |
| Database | PostgreSQL |

---

## Key Features

- **Feature Engineering** — 5+ technical indicators computed from raw OHLCV data: RSI, MACD, moving averages, volatility, and momentum
- **Time-Series Preprocessing** — Data normalization, sequence generation, and train/validation/test split with no data leakage
- **LSTM Architecture** — Multi-layer LSTM with dropout regularization, EarlyStopping callback, and threshold optimization via ROC curve analysis
- **Automated Backtesting** — Backtrader-integrated strategy that executes trades from LSTM predictions and reports Sharpe ratio, max drawdown, and cumulative returns
- **Portfolio Dashboard** — React frontend displaying live KPIs: Sharpe ratio, drawdown, and return metrics

---

## Project Structure

```
finance-ai/
├── data/
│   └── fetch_data.py          # yfinance API data ingestion + PostgreSQL storage
├── features/
│   └── engineering.py         # RSI, MACD, volatility, momentum feature construction
├── model/
│   ├── lstm_model.py          # LSTM architecture, training loop, EarlyStopping
│   └── evaluate.py            # ROC curve analysis, threshold optimization, metrics
├── backtest/
│   └── strategy.py            # Backtrader strategy using LSTM predictions
├── dashboard/                 # React frontend for portfolio KPI visualization
├── requirements.txt
└── README.md
```

---

## Results

| Metric | Result |
|---|---|
| Prediction Task | Directional movement (up/down) |
| Validation Method | ROC curve analysis + Sharpe ratio vs. buy-and-hold |
| Regularization | Dropout + EarlyStopping + threshold optimization |
| Backtesting | Automated via Backtrader — Sharpe ratio, drawdown, return reporting |

---

## Setup & Usage

```bash
# Clone the repository
git clone https://github.com/mosdeffy/finance-ai.git
cd finance-ai

# Install dependencies
pip install -r requirements.txt

# Fetch and store stock data
python data/fetch_data.py

# Engineer features
python features/engineering.py

# Train the LSTM model
python model/lstm_model.py

# Evaluate model performance
python model/evaluate.py

# Run backtesting strategy
python backtest/strategy.py
```

---

## Author

**Yanni Aguilar**
[LinkedIn](https://linkedin.com/in/yanni-aguilar) | [GitHub](https://github.com/mosdeffy)

*Georgia Institute of Technology — B.S. Industrial Engineering, Data Science & Analytics (GPA: 4.0)*
*Kennesaw State University — B.S. Data Science & Analytics (GPA: 4.0)*
