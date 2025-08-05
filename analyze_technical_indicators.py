import pandas as pd
import numpy as np
import ta

from eda.plots import plot_correlation_matrix


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Momentum
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['stoch_k'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    df['stoch_d'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch_signal()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['macd_signal'] = ta.trend.MACD(close=df['close']).macd_signal()

    # Volatility
    df['bollinger_h'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_l'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

    # Trend
    df['ema_10'] = ta.trend.EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()

    return df


# === Example Usage ===
if __name__ == '__main__':
    df = pd.read_csv('data/gold_features.csv')
    plot_correlation_matrix(df)
