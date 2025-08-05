# eda/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True)
    st.pyplot(plt.gcf())
    plt.clf()


def plot_return_distribution(df):
    plt.figure(figsize=(10, 4))
    sns.histplot(df["future_return"].dropna(), bins=100, kde=True, color="gold")
    plt.title("Distribution of Future Returns")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_signal_frequencies(df):
    signal_cols = [col for col in df.columns if col.startswith("signal_")]
    if not signal_cols:
        st.warning("No signal columns found.")
        return
    counts = df[signal_cols].sum().sort_values()
    counts.plot(kind="barh", figsize=(8, 5), color="teal")
    plt.title("Signal Frequencies")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_feature_distributions(df, features):
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(8, 3))
            sns.histplot(df[feature].dropna(), bins=50, kde=True)
            plt.title(f"Distribution of {feature}")
            st.pyplot(plt.gcf())
            plt.clf()


def plot_confusion_matrix(df, y_true_col, y_pred_col):
    y_true = df[y_true_col].dropna()
    y_pred = df[y_pred_col].dropna()

    # Align both series
    df_valid = df[[y_true_col, y_pred_col]].dropna()
    cm = confusion_matrix(df_valid[y_true_col], df_valid[y_pred_col])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_feature_by_label(df, label_col, feature_col):
    if label_col not in df.columns:
        st.error(f"Label column '{label_col}' not found in DataFrame.")
        st.write("Available columns:", df.columns.tolist())
        return
    sns.boxplot(x=label_col, y=feature_col, data=df)
    plt.title(f"{feature_col} by {label_col}")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_signal_label_performance(df, signal_col, label_col):
    if signal_col not in df.columns or label_col not in df.columns:
        st.warning(f"Missing {signal_col} or {label_col}")
        return
    grouped = df[df[signal_col] == 1].groupby(label_col)["future_return"].agg(["mean", "count"])
    st.write(f"Performance of {signal_col} by {label_col}")
    st.dataframe(grouped)


def plot_calendar_heatmap(df, date_col, value_col):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = df.copy()

    # Fix: handle timezone-aware datetimes
    df["date_only"] = pd.to_datetime(df[date_col], utc=True).dt.date

    df["year"] = pd.to_datetime(df["date_only"]).dt.year
    df["month"] = pd.to_datetime(df["date_only"]).dt.month
    df["day"] = pd.to_datetime(df["date_only"]).dt.day

    pivot = df.pivot_table(index="month", columns="day", values=value_col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap="RdYlGn", ax=ax, annot=False, cbar_kws={'label': value_col})
    ax.set_title(f"Calendar Heatmap: {value_col}")
    st.pyplot(fig)


def plot_trend_vs_return(df, trend_col="ema_trend", return_col="future_return"):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=trend_col, y=return_col, data=df)
    plt.title(f"{return_col} by {trend_col}")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_label_distribution(df, label_col="label"):
    plt.figure(figsize=(6, 4))
    df[label_col].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {label_col}")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_distribution_by_label(df, label_col, feature):
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x=feature, hue=label_col)
    plt.title(f"{feature} Distribution by {label_col}")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_volatility_label_scatter(df, vol_col="atr_14", return_col="future_return", label_col="label"):
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x=vol_col, y=return_col, hue=label_col)
    plt.title(f"{return_col} vs {vol_col} by {label_col}")
    st.pyplot(plt.gcf())
    plt.clf()
