import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Signal Evaluation Function with Improvements ===
def evaluate_signal(df: pd.DataFrame, signal_col: str, target_col: str = 'future_return_class', regime_filter: str = None):
    """
    Evaluates how well a signal predicts the future return class.
    Supports market regime filtering and logs performance.
    """
    results = {}

    # Optional regime filtering
    if regime_filter:
        df = df[df['market_regime'] == regime_filter]

    # Drop missing values
    valid = df[[signal_col, target_col]].dropna()
    signal = valid[signal_col]
    target = valid[target_col]

    # Keep only active signals
    active = signal != 0
    if active.sum() == 0:
        print(f"\n⚠️ No active signals in '{signal_col}' (after regime filter: {regime_filter})")
        return None

    y_true = target[active]
    y_pred = signal[active]

    # === Metrics ===
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Short (-1)', 'Neutral (0)', 'Long (+1)'])
    disp.plot(cmap='Blues')
    plt.title(f"{signal_col} - Confusion Matrix")
    plt.grid(False)
    plt.show()

    # === Signal Frequency Plot ===
    signal_counts = y_pred.value_counts().sort_index()
    signal_counts.plot(kind='bar', title=f'{signal_col} Signal Frequency', color='skyblue')
    plt.xlabel('Signal Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Logging ===
    print(f"\n🔍 Evaluation for: '{signal_col}'")
    print(f"✅ Accuracy: {accuracy:.2f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    results['signal'] = signal_col
    results['accuracy'] = accuracy
    results['n_signals'] = active.sum()
    results.update({f"{k}_{m}": v for k, metrics in report_dict.items() if k in ['-1', '0', '1']
                    for m, v in metrics.items()})

    return results


# === Apply Evaluation to All Signals and Save to DataFrame ===
def evaluate_all_signals(df: pd.DataFrame, regime_filter: str = None) -> pd.DataFrame:
    print("📊 Starting signal evaluation...\n")
    results = []

    for signal_col in signal_columns:
        if signal_col in df.columns:
            result = evaluate_signal(df, signal_col=signal_col, regime_filter=regime_filter)
            if result:
                results.append(result)
        else:
            print(f"❌ Column '{signal_col}' not found in DataFrame")

    # Return as DataFrame
    results_df = pd.DataFrame(results)
    print("\n📋 Summary Table:")
    print(results_df[['signal', 'accuracy', 'n_signals']])
    return results_df


# === Signal Columns to Evaluate ===
signal_columns = [
    'breakout_signal',
    'pullback_signal',
    'volatility_spike',
    'range_expansion',
    'price_compression',
    'price_acceleration',
    'volume_spike'
]

# === Usage ===
if __name__ == "__main__":
    df = pd.read_csv('data/gold_features.csv')
    results_df = evaluate_all_signals(df)  # optionally: regime_filter="trend"
    results_df.to_csv("output/signal_evaluation_results.csv", index=False)
