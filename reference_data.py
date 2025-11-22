# save_reference.py
import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta

# Configuration (Must match your notebook)
TARGET_TICKER = 'TSLA'
START_DATE = '2020-01-01' 
# We fetch data up to yesterday to simulate the training set
END_DATE = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

def feature_engineering(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Handle zero division
    df['Close'] = df['Close'].replace(0, 0.0001)
    df['Open'] = df['Open'].replace(0, 0.0001)

    df['Volatility'] = (df['High'] - df['Low']) / df['Close']
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['SMA'] = ta.trend.SMAIndicator(df['Close'], window=21).sma_indicator()
    df['Return'] = (df['Close'] - df['Open']) / df['Open']
    # Use Volume directly
    return df[['Volatility', 'RSI', 'SMA', 'Return', 'Volume']]

print("Downloading Reference Data...")
data_target = yf.download(TARGET_TICKER, start=START_DATE, end=END_DATE, progress=False)
data_spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)

df_target = feature_engineering(data_target)
df_spy = feature_engineering(data_spy)

# Merge
reference_df = pd.merge(df_target, df_spy, left_index=True, right_index=True, suffixes=('_Target', '_SPY'))
reference_df = reference_df.dropna()

# Save to CSV for the monitoring system
reference_df.to_csv('reference_data.csv', index=False)
print(f"Reference data saved: {reference_df.shape}")