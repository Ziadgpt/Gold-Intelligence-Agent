import yfinance as yf
import pandas as pd
import numpy as np
import os
from features.technical_indicators import add_technical_indicators
from eda.plots import plot_correlation_matrix
# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# === Fetch OHLCV Gold Data ===
def fetch_gold_data():
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="max", interval="1d", auto_adjust=False)

    # Ensure OHLCV structure
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df.columns = [col.lower() for col in df.columns]  # lowercase all column names
    df.index.name = 'date'
    return df

# === Price-Based Feature Engineering ===
def add_price_features(df):
    df = df.copy().reset_index()

    # Basic returns
    df['pct_change'] = df['close'].pct_change()

    # Breakout: price > previous 20-day high
    df['20d_high'] = df['high'].rolling(window=20).max()
    df['breakout'] = df['high'] > df['20d_high'].shift(1)

    # Simple trend (MA crossover)
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['trend_up'] = (df['ma10'] > df['ma20']).astype(int)

    # Range expansion
    df['daily_range'] = df['high'] - df['low']
    df['avg_range'] = df['daily_range'].rolling(window=20).mean()
    df['range_expansion'] = (df['daily_range'] > df['avg_range']).astype(int)

    # Volatility
    df['rolling_std_10'] = df['close'].rolling(window=10).std()
    df['rolling_std_20'] = df['close'].rolling(window=20).std()

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    # Skewness & Kurtosis
    df['skew_20'] = df['close'].rolling(window=20).skew()
    df['kurtosis_20'] = df['close'].rolling(window=20).kurt()

    # Date-related features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # Future return + position labeling
    df['future_close_1d'] = df['close'].shift(-1)
    df['future_return_1d'] = (df['future_close_1d'] - df['close']) / df['close']
    df['position_1d'] = np.where(df['future_return_1d'] > 0.005, 1,
                          np.where(df['future_return_1d'] < -0.005, -1, 0))

    return df.drop(columns=['20d_high', 'ma10', 'ma20', 'avg_range'])

# === Main Runner ===
if __name__ == "__main__":
    df = fetch_gold_data()
    df = add_price_features(df)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    df.to_csv("data/gold_features.csv", index=False)
    print("✅ Data saved to data/gold_features.csv")
