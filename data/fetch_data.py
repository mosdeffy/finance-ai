"""
fetch_data.py
Fetches historical OHLCV stock data via the yfinance API and stores it in PostgreSQL.
"""
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost/finance_ai")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker from yfinance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df["Ticker"] = ticker
    df.reset_index(inplace=True)
    return df


def store_to_postgres(df: pd.DataFrame, table_name: str, engine) -> None:
    """Write DataFrame to PostgreSQL via SQLAlchemy."""
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Stored {len(df)} rows to '{table_name}'.")


def main():
    engine = create_engine(DB_URL)
    all_data = []
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        df = fetch_ohlcv(ticker, START_DATE, END_DATE)
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    store_to_postgres(combined, "ohlcv_data", engine)
    print(f"Done. Total rows stored: {len(combined)}")


if __name__ == "__main__":
    main()
