import numpy as np
import pandas as pd
import yfinance as yf
import ta
from flask import Flask, request, render_template
import tensorflow as tf
import logging
import joblib
from datetime import datetime, timedelta
import csv
import os

app = Flask(__name__)

# Setup Logging (System logs)
logging.basicConfig(filename='system.log', level=logging.INFO)

# --- CONFIGURATION ---
# Only allow these tickers. Everything else will be ignored.
ALLOWED_TICKERS = ['TSLA', 'SPY', 'AAPL', 'NVDA', 'AMZN']

# Production Data Log (For Model Monitoring)
PROD_LOG_FILE = 'production_logs.csv'
FEATURE_COLS = [
    'Volatility_Target', 'RSI_Target', 'SMA_Target', 'Return_Target', 'Volume_Target',
    'Volatility_SPY', 'RSI_SPY', 'SMA_SPY', 'Return_SPY', 'Volume_SPY'
]

# Initialize CSV header if file doesn't exist
if not os.path.exists(PROD_LOG_FILE):
    with open(PROD_LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_COLS + ['Prediction_Score', 'Prediction_Class', 'Timestamp'])

# Load Model and Scaler
print("Loading resources...")
model = tf.keras.models.load_model('final_model.keras')
scaler = joblib.load('scaler.pkl')

input_shape = model.input_shape
NEEDS_SEQUENCE = len(input_shape) == 3 
SEQ_LENGTH = input_shape[1] if NEEDS_SEQUENCE else 1

def get_latest_features(ticker):
    """Fetches data and returns both Scaled Input (for model) and Raw Features (for logging)"""
    lookback = 60 + SEQ_LENGTH
    start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
    
    data_target = yf.download(ticker, start=start_date, progress=False)
    data_spy = yf.download('SPY', start=start_date, progress=False)

    if len(data_target) < (22 + SEQ_LENGTH):
        raise ValueError("Not enough recent data.")

    def engineer(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df['Volatility'] = (df['High'] - df['Low']) / df['Close']
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['SMA'] = ta.trend.SMAIndicator(df['Close'], window=21).sma_indicator()
        df['Return'] = (df['Close'] - df['Open']) / df['Open']
        return df[['Volatility', 'RSI', 'SMA', 'Return', 'Volume']]

    df_target = engineer(data_target)
    df_spy = engineer(data_spy)

    merged_df = pd.merge(df_target, df_spy, left_index=True, right_index=True, suffixes=('_Target', '_SPY'))
    
    # Get Raw Features (Last row) for Logging
    # Note: We monitor the "Raw" features before scaling to detect real-world market shifts
    raw_features_log = merged_df.iloc[-1].tolist()

    # Process for Model
    recent_data = merged_df.iloc[-SEQ_LENGTH:].values
    scaled_data = scaler.transform(recent_data)
    
    if NEEDS_SEQUENCE:
        final_input = np.array([scaled_data])
    else:
        final_input = np.array([scaled_data[-1]])
    
    return final_input, raw_features_log

def log_to_csv(features, score, pred_class):
    """Logs data for Drift Monitoring"""
    with open(PROD_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(features + [score, pred_class, timestamp])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get ticker, strip whitespace, force uppercase
    ticker = request.form['ticker'].upper().strip()
    
    # --- VALIDATION STEP ---
    # If input is not in our allowed list, fail silently (show nothing)
    if ticker not in ALLOWED_TICKERS:
        logging.warning(f"Ignored unsupported input: {ticker}")
        return render_template('index.html')

    try:
        # 1. Get Features
        model_input, raw_features = get_latest_features(ticker)
        
        # 2. Predict
        prediction = model.predict(model_input)
        raw_score = float(prediction[0][0])
        pred_class = 1 if raw_score > 0.5 else 0
        result_text = "HIGH RISK" if pred_class == 1 else "NORMAL"
        
        # 3. Log for Monitoring
        log_to_csv(raw_features, raw_score, pred_class)
        logging.info(f"Ticker: {ticker} | Score: {raw_score}")

        return render_template('index.html', 
                             prediction_text=f'Prediction for {ticker}: {result_text}',
                             score_text=f'(Confidence Score: {raw_score:.4f})')

    except Exception as e:
        logging.error(f"Error predicting {ticker}: {str(e)}")
        # For real errors (like yfinance failure), we also fail silently to be safe
        # or you can show specific text if preferred.
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
