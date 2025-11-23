"""
train_model.py

Retrains the TSLA vs SPY volatility classifier (FNN) using the same
feature logic as the deployed Flask app (app.py).

Outputs:
- final_model.keras        (used by app.py)
- scaler.pkl               (used by app.py)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import ta

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# ---------------- CONFIG ----------------
TARGET_TICKER = "TSLA"
BENCHMARK_TICKER = "SPY"
START_DATE = "2010-01-01"

VOLATILITY_THRESHOLD_PERCENTILE = 0.70  # top 30% = high volatility
TEST_SIZE = 0.15                        # test share
VAL_SIZE = 0.15                         # from remaining train

MODEL_PATH = "best_model/final_model.keras"
SCALER_PATH = "best_model/scaler.pkl"

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------- DATA PIPELINE ------------

def download_data(ticker: str, start_date: str) -> pd.DataFrame:
    print(f"[DATA] Downloading data for {ticker} since {start_date}...")
    data = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    data.index = pd.to_datetime(data.index)
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same logic as app.py -> get_latest_features():
    Volatility, RSI(14), SMA(21), Return, Volume
    """
    # Drop multi-index if present (like in app.py)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.copy()
    df["Volatility"] = (df["High"] - df["Low"]) / df["Close"]
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["SMA"] = ta.trend.SMAIndicator(df["Close"], window=21).sma_indicator()
    df["Return"] = (df["Close"] - df["Open"]) / df["Open"]

    # final feature set
    return df[["Volatility", "RSI", "SMA", "Return", "Volume"]]


def build_merged_feature_frame() -> pd.DataFrame:
    """Download TSLA + SPY and build a merged feature dataframe."""
    tsla_raw = download_data(TARGET_TICKER, START_DATE)
    spy_raw = download_data(BENCHMARK_TICKER, START_DATE)

    tsla_feat = engineer_features(tsla_raw)
    spy_feat = engineer_features(spy_raw)

    # suffixes consistent with app.py ( _Target and _SPY )
    merged = pd.merge(
        tsla_feat,
        spy_feat,
        left_index=True,
        right_index=True,
        suffixes=("_Target", "_SPY"),
    )

    # Drop rows with any NaNs (from RSI/SMA warmup)
    merged = merged.dropna()

    return merged


def create_target_variable(df: pd.DataFrame, threshold_percentile: float) -> pd.DataFrame:
    """
    Binary target = 1 if Volatility_Target is above percentile threshold.
    Shift by -1 day so features predict *next* day's volatility.
    """
    df = df.copy()
    vol = df["Volatility_Target"]
    threshold = vol.quantile(threshold_percentile)
    print(f"[LABEL] Volatility threshold ({threshold_percentile*100:.1f}th percentile): {threshold:.4f}")

    df["Target"] = (vol > threshold).astype(int)

    # shift to make it "predict tomorrow"
    df["Target"] = df["Target"].shift(-1)
    df = df.dropna(subset=["Target"])

    df["Target"] = df["Target"].astype(int)
    return df


def load_and_preprocess_data():
    """
    Full data pipeline:
      - build features for TSLA + SPY
      - create binary target
      - split train/val/test
      - fit scaler on train and transform all splits
    """
    df = build_merged_feature_frame()
    df = create_target_variable(df, VOLATILITY_THRESHOLD_PERCENTILE)

    feature_cols = [c for c in df.columns if c != "Target"]

    X = df[feature_cols].values
    y = df["Target"].values

    # first split off test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_SEED, stratify=y
    )

    # then split train_full into train + val
    val_size_adj = VAL_SIZE / (1.0 - TEST_SIZE)  # adjust for already-split test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size_adj,
        shuffle=True,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    print(f"[SHAPE] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # scale features
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_cols,
    )


# ------------- MODEL (FNN) ----------------

def build_ffn(input_dim: int) -> tf.keras.Model:
    """Feedforward NN, same as notebook's best model."""
    model = models.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_and_save():
    print("[INFO] Starting retraining pipeline (FNN)...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_cols,
    ) = load_and_preprocess_data()

    # Build model
    model = build_ffn(input_dim=X_train.shape[1])

    # Callbacks (TensorBoard optional)
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print("[INFO] Training FNN model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[tb_cb, es_cb],
        verbose=1,
    )

    print("[INFO] Evaluating on test set...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test Loss={test_loss:.4f}, Acc={test_acc:.4f}, AUC={test_auc:.4f}")

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"[METRIC] Test ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Save model and scaler where app.py expects them
    print(f"[SAVE] Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)

    print(f"[SAVE] Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)

    print("[SUCCESS] Retraining complete. Model + scaler updated.")


if __name__ == "__main__":
    train_and_save()
