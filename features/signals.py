import pandas as pd
import numpy as np


# === Signal 1: Breakout Detection ===
def detect_breakout(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Detects significant breakouts from recent high/low range.
    +1 = upside breakout, -1 = downside breakout, 0 = no breakout.
    """
    recent_high = df['High'].rolling(window=window).max().shift(1)
    recent_low = df['Low'].rolling(window=window).min().shift(1)

    signal = pd.Series(0, index=df.index)
    signal[df['Close'] > recent_high] = 1
    signal[df['Close'] < recent_low] = -1
    return signal


# === Signal 2: Pullback Detection ===
def detect_pullback(df: pd.DataFrame, ma_window: int = 50, lookback: int = 20) -> pd.Series:
    """
    Detects significant pullbacks during trends based on average historical pullback depth.
    Signals 1 if current pullback exceeds average of past pullbacks during same trend direction.
    """
    df = df.copy()
    ma = df['Close'].rolling(window=ma_window).mean()
    price = df['Close']

    df['trend'] = np.where(price > ma, 1, np.where(price < ma, -1, 0))

    pullback_up = (ma - price) / ma  # uptrend
    pullback_down = (price - ma) / ma  # downtrend

    df['pullback_depth'] = 0.0
    df.loc[df['trend'] == 1, 'pullback_depth'] = pullback_up
    df.loc[df['trend'] == -1, 'pullback_depth'] = pullback_down

    avg_pullback = df['pullback_depth'].rolling(window=lookback).mean()

    df['pullback_signal'] = (
            (df['pullback_depth'] > avg_pullback) & (df['trend'] != 0)
    ).astype(int)

    return df['pullback_signal']


# === Signal 3: Volatility Spike Detection ===
def detect_volatility_spike(df: pd.DataFrame, window: int = 14, threshold: float = 2.0) -> pd.Series:
    """
    Detects volatility spikes: 1 if rolling std > mean + N * std of std.
    """
    rolling_std = df['Close'].rolling(window=window).std()
    baseline_mean = rolling_std.rolling(window).mean()
    baseline_std = rolling_std.rolling(window).std()
    return (rolling_std > (baseline_mean + threshold * baseline_std)).astype(int)


# === Signal 4: Range Expansion Detection ===
def detect_range_expansion(df: pd.DataFrame, window: int = 20, z_thresh: float = 2.0,
                           return_score: bool = False) -> pd.Series:
    """
    Detects large range bars based on z-score of (High - Low).
    If return_score=True, returns the z-score (feature).
    """
    daily_range = df['High'] - df['Low']
    mean_range = daily_range.rolling(window).mean()
    std_range = daily_range.rolling(window).std()
    z_score = (daily_range - mean_range) / std_range

    if return_score:
        return z_score
    else:
        return (z_score > z_thresh).astype(int)


# === Signal 5: Price Compression (Low Volatility) ===
def detect_price_compression(df: pd.DataFrame, window: int = 20, quantile: float = 0.2) -> pd.Series:
    """
    Detects price compression: rolling std below historical quantile.
    """
    vol = df['Close'].rolling(window).std()
    low_vol_threshold = vol.rolling(window).quantile(quantile)
    return (vol < low_vol_threshold).astype(int)


# === Signal 6: Price Acceleration ===
def detect_price_acceleration(df: pd.DataFrame, window: int = 10, threshold: float = 2.0) -> pd.Series:
    """
    Detects sudden moves (spikes) in price: return > N * rolling std of returns.
    """
    returns = df['Close'].pct_change()
    vol = returns.rolling(window).std()
    return (returns > (threshold * vol)).astype(int)


# === Signal 7: Volume Spike ===
def detect_volume_spike(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
    """
    Detects unusual volume activity: volume > mean + N * std.
    """
    vol = df['Volume']
    vol_mean = vol.rolling(window).mean()
    vol_std = vol.rolling(window).std()
    return (vol > (vol_mean + threshold * vol_std)).astype(int)


# === Apply All Signals Wrapper ===
def apply_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all signal functions and returns updated dataframe with signal columns.
    """
    df = df.copy()
    df['breakout_signal'] = detect_breakout(df)
    df['pullback_signal'] = detect_pullback(df)
    df['volatility_spike'] = detect_volatility_spike(df)
    df['range_expansion'] = detect_range_expansion(df)
    df['price_compression'] = detect_price_compression(df)
    df['price_acceleration'] = detect_price_acceleration(df)
    df['volume_spike'] = detect_volume_spike(df)
    return df
