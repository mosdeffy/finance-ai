"""
lstm_model.py
Builds and trains a multi-layer LSTM neural network for directional stock
price movement prediction with EarlyStopping and dropout regularization.
No data leakage: train/validation/test split is time-ordered.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import joblib

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost/finance_ai")
SEQ_LEN = 30
FEATURES = ["Close", "RSI", "MACD", "MACD_Signal", "Volatility", "Momentum", "MA_10", "MA_20"]
TARGET = "Target"
MODEL_PATH = "model/saved_model.keras"
SCALER_PATH = "model/scaler.pkl"


def build_sequences(data: np.ndarray, seq_len: int):
    """Convert flat time-series array into overlapping sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len, :-1])
        y.append(data[i + seq_len - 1, -1])
    return np.array(X), np.array(y)


def build_model(input_shape: tuple) -> Sequential:
    """Define multi-layer LSTM architecture with dropout regularization."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM features", engine)

    data = df[FEATURES + [TARGET]].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    X, y = build_sequences(data_scaled, SEQ_LEN)

    # Time-ordered split to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_model((SEQ_LEN, len(FEATURES)))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    np.save("model/X_test.npy", X_test)
    np.save("model/y_test.npy", y_test)
    print("Test arrays saved for evaluation.")


if __name__ == "__main__":
    main()
