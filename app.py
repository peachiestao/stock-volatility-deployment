import numpy as np
import pandas as pd
import yfinance as yf
import ta
from flask import Flask, request, render_template
import tensorflow as tf
import logging
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Setup Logging
logging.basicConfig(filename='model_monitor.log', level=logging.INFO)

# Load Model and Scaler
print("Loading resources...")
model = tf.keras.models.load_model('final_model.keras')
scaler = joblib.load('scaler.pkl')

# --- SMART SHAPE DETECTION ---
# Check if the model expects a sequence (LSTM/CNN) or flat data (FFN)
input_shape = model.input_shape
NEEDS_SEQUENCE = len(input_shape) == 3 
SEQ_LENGTH = input_shape[1] if NEEDS_SEQUENCE else 1

print(f"Model loaded! Expects sequences: {NEEDS_SEQUENCE} (Length: {SEQ_LENGTH})")

# --- CONFIGURATION ---
# Only these tickers will trigger a prediction.
# "AAPL" or "Two" will be ignored.
ALLOWED_TICKERS = ['TSLA', 'SPY']

def get_latest_features(ticker):
    """
    Fetches data, calculates features, scales them, and reshapes for the model.
    """
    # Fetch enough data for 21-day SMA + Sequence Length
    lookback = 60 + SEQ_LENGTH
    start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
    
    # 1. Download Data
    data_target = yf.download(ticker, start=start_date, progress=False)
    data_spy = yf.download('SPY', start=start_date, progress=False)

    if len(data_target) < (22 + SEQ_LENGTH) or len(data_spy) < (22 + SEQ_LENGTH):
        raise ValueError("Not enough recent data found for prediction.")

    # 2. Feature Engineering Function
    def engineer(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df['Volatility'] = (df['High'] - df['Low']) / df['Close']
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['SMA'] = ta.trend.SMAIndicator(df['Close'], window=21).sma_indicator()
        df['Return'] = (df['Close'] - df['Open']) / df['Open']
        return df[['Volatility', 'RSI', 'SMA', 'Return', 'Volume']]

    # 3. Calculate All Features
    df_target = engineer(data_target)
    df_spy = engineer(data_spy)

    # 4. Merge & Select Data
    merged_df = pd.merge(df_target, df_spy, left_index=True, right_index=True, suffixes=('_Target', '_SPY'))
    recent_data = merged_df.iloc[-SEQ_LENGTH:].values

    # 5. Scale & Reshape
    scaled_data = scaler.transform(recent_data)
    
    if NEEDS_SEQUENCE:
        final_input = np.array([scaled_data])
    else:
        final_input = np.array([scaled_data[-1]])
    
    return final_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input, remove whitespace, force uppercase
    # "appl " becomes "APPL"
    ticker = request.form['ticker'].upper().strip()
    
    # --- VALIDATION STEP ---
    # If input is NOT 'TSLA' or 'SPY', we stop here.
    # We return the template with NO variables, so the result area is empty.
    if ticker not in ALLOWED_TICKERS:
        logging.warning(f"Ignored unsupported input: {ticker}")
        return render_template('index.html') 

    try:
        # 1. Get Smart Features
        input_features = get_latest_features(ticker)
        
        # 2. Predict
        prediction = model.predict(input_features)
        raw_score = float(prediction[0][0])
        
        # 3. Interpret
        result_text = "HIGH Volatility Risk" if raw_score > 0.5 else "Normal Volatility"
        
        logging.info(f"Ticker: {ticker} | Score: {raw_score}")

        return render_template('index.html', 
                             prediction_text=f'Prediction for {ticker}: {result_text}',
                             score_text=f'(Confidence Score: {raw_score:.4f})')

    except Exception as e:
        # On any error (e.g., internet issue), also show nothing to the user
        logging.error(f"Error predicting {ticker}: {str(e)}")
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
