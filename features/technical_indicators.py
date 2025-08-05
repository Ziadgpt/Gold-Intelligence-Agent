import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import analyze_technical_indicators
from analyze_technical_indicators import add_technical_indicators


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/gold_features.csv", parse_dates=["date"])
    return df

# Plot Correlation Matrix
def plot_correlation_matrix(df, target_col="future_return"):
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

# App title
st.set_page_config(page_title="Gold Signal Lab", layout="wide")
st.title("🏛️ Gold Feature Engineering Dashboard")

# Load data
df = load_data()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🔍 Patterns", "📅 Calendar View", "📊 Statistics", "🧪 Technical Lab"
])

# === Tab 1: Overview ===
with tab1:
    st.subheader("Raw Data Snapshot")
    st.dataframe(df.tail(100))

    st.subheader("Feature Overview")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    st.write("Missing values per column:")
    st.dataframe(df.isnull().sum())

# === Tab 2: Patterns ===
with tab2:
    st.subheader("Correlation with Future Return")
    plot_correlation_matrix(df)

# === Tab 3: Calendar View ===
with tab3:
    st.subheader("Calendar View by Return Buckets")

    df["date"] = pd.to_datetime(df["date"])
    df["return_bucket"] = pd.qcut(df["future_return"], 3, labels=["Low", "Medium", "High"])
    calendar_df = df[["date", "return_bucket"]].copy()

    st.write("Return Bucket Frequency:")
    st.bar_chart(calendar_df["return_bucket"].value_counts())

# === Tab 4: Statistics ===
with tab4:
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

    st.subheader("Distribution of Target (Future Return)")
    fig, ax = plt.subplots()
    sns.histplot(df["future_return"], bins=50, kde=True, ax=ax)
    ax.set_title("Future Return Distribution")
    st.pyplot(fig)

# === Tab 5: Technical Lab ===
with tab5:
    st.subheader("Technical Indicators Lab")

    df_ta = add_technical_indicators(df)

    selected_indicator = st.selectbox(
        "Select Technical Indicator to Plot",
        ["rsi_14", "macd", "macd_signal", "bollinger_width", "atr_14", "ema_20", "ema_50", "stoch_k", "stoch_d"]
    )

    st.line_chart(df_ta.set_index("date")[selected_indicator])

    st.markdown("""
        Use this lab to explore the behavior of technical indicators across time. 
        You can combine this later with overlays (e.g., price) or signal flags.
    """)
