import streamlit as st
import pandas as pd
import numpy as np
import os

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
from analyze_technical_indicators import add_technical_indicators
from eda.plots import (
    plot_correlation_matrix,
    plot_return_distribution,
    plot_confusion_matrix,
    plot_signal_label_performance,
    plot_calendar_heatmap,
    plot_trend_vs_return,
    plot_label_distribution,
    plot_distribution_by_label,
    plot_feature_by_label,
    plot_volatility_label_scatter,
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/gold_features.csv", parse_dates=["date"])
    return df

# Config
st.set_page_config(page_title="Gold Signal Lab", layout="wide")
st.title("🏛️ Gold Trading Feature Explorer")

df = load_data()

# Ensure future_return exists
if "future_return" not in df.columns:
    df["future_return"] = df["close"].pct_change().shift(-1)

# Tabs
tabs = st.tabs([
    "📊 Overview", "📈 Correlations", "📅 Calendar", "🔬 Labels",
    "📡 Signals", "🧪 Technical Lab", "⚖️ Confusion Matrix"
])

# === 1. Overview ===
with tabs[0]:
    st.subheader("Raw Data Preview")
    st.dataframe(df.tail(100))

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Return Distribution")
    plot_return_distribution(df)

# === 2. Correlations ===
with tabs[1]:
    st.subheader("📘 Math/Time Features Correlation")
    math_cols = [col for col in df.columns if any(key in col for key in ["std", "skew", "kurt", "hour", "weekday", "month"])]
    plot_correlation_matrix(df[math_cols])

    st.subheader("📗 Technical Indicators Correlation")
    ta_cols = [col for col in df.columns if col.startswith(("rsi", "macd", "bollinger", "atr", "stoch", "ema"))]
    plot_correlation_matrix(df[ta_cols])

# === 3. Calendar ===
with tabs[2]:
    st.subheader("Calendar Return View")
    if "future_return" in df.columns:
        plot_calendar_heatmap(df, "date", "future_return")
    else:
        st.warning("Missing `future_return` column.")

# === 4. Label Analysis ===
with tabs[3]:
    st.subheader("Label Distribution")
    label_col = st.selectbox("Select label column", ["label", "return_bucket", "predicted_class"])
    if label_col in df.columns:
        plot_label_distribution(df, label_col)

    st.subheader("Feature by Label")
    selected_features = st.multiselect("Select features", df.select_dtypes("number").columns.tolist(), default=["future_return"])
    st.write("Columns in df:", df.columns.tolist())
    st.write("Missing 'label'?", 'label' not in df.columns)
    for feat in selected_features:
        plot_feature_by_label(df, label_col, feat)

# === 5. Signal Behavior ===
with tabs[4]:
    st.subheader("Signal Performance by Label")
    signal_cols = [col for col in df.columns if col.startswith("signal_")]
    for signal in signal_cols:
        plot_signal_label_performance(df, signal, label_col)

# === 6. Technical Lab ===
with tabs[5]:
    st.subheader("Technical Indicator Explorer")
    df_ta = add_technical_indicators(df.copy())

    ta_indicators = [
        "rsi_14", "macd", "macd_signal", "bollinger_width",
        "atr_14", "ema_20", "ema_50", "stoch_k", "stoch_d"
    ]
    selected_indicator = st.selectbox("Select Indicator", ta_indicators)
    st.line_chart(df_ta.set_index("date")[selected_indicator])

    st.markdown("Use this to study time-series behavior of individual indicators.")

# === 7. Confusion Matrix ===
with tabs[6]:
    st.subheader("Confusion Matrix")
    if "true_class" in df.columns and "predicted_class" in df.columns:
        plot_confusion_matrix(df, y_true_col="true_class", y_pred_col="predicted_class")
    else:
        st.warning("Add `true_class` and `predicted_class` columns to view confusion matrix.")
