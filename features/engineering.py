"""
engineering.py
Computes technical indicators (RSI, MACD, moving averages, volatility, momentum)
from raw OHLCV data and saves the enriched feature set for model training.
"""
import pandas as pd
import numpy as np
import ta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost/finance_ai")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to an OHLCV DataFrame."""
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Momentum indicators
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # Trend indicators
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Volatility
    df["Volatility"] = df["Close"].pct_change().rolling(14).std()

    # Momentum (rate of change)
    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    # Moving averages
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # Target: 1 if next-day close is higher, 0 otherwise
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


def main():
    engine = create_engine(DB_URL)
    raw = pd.read_sql("SELECT * FROM ohlcv_data", engine)
    enriched = raw.groupby("Ticker", group_keys=False).apply(engineer_features)
    enriched.to_sql("features", engine, if_exists="replace", index=False)
    print(f"Feature engineering complete. Shape: {enriched.shape}")


if __name__ == "__main__":
    main()
