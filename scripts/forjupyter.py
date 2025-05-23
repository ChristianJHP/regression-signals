

"""
Crypto ML Pipeline Summary and Visualization
--------------------------------------------
This script provides a comprehensive walkthrough of the machine learning workflow
used to analyze and predict crypto price movements.

Goals:
- Train regression models to forecast price
- Train classifiers to detect high-confidence pump signals
- Evaluate model accuracy, directional correctness, and trading feasibility
- Explore potential alpha and improve model reliability
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# Define paths
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
price_path = os.path.join(project_root, "data", "crypto_ohlcv.csv")
signal_dir = os.path.join(project_root, "data", "signals")

print("Loading main OHLCV dataset...")
df = pd.read_csv(price_path, parse_dates=["timestamp"])
print(f"Loaded {len(df)} rows from OHLCV data.")

print("\nPreview of market data:")
print(df.head())

print("\nLoading signal files for each coin...")
signal_files = [f for f in os.listdir(signal_dir) if f.endswith("_signals.csv")]
summary = []

for file in signal_files:
    path = os.path.join(signal_dir, file)
    symbol = file.split("_")[0]
    df_signal = pd.read_csv(path)
    max_conf_row = df_signal.loc[df_signal['confidence'].idxmax()]

    summary.append({
        'symbol': symbol,
        'timestamp': max_conf_row['timestamp'],
        'confidence': max_conf_row['confidence'],
        'predicted_label': max_conf_row['predicted'],
        'actual_label': max_conf_row.get('actual', 'N/A')
    })

summary_df = pd.DataFrame(summary).sort_values(by='confidence', ascending=False)
print("\nTop signal predictions by confidence:")
print(summary_df)

# Plot example: Confidence distribution
plt.figure(figsize=(10, 5))
plt.hist(summary_df['confidence'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Top Signal Confidences")
plt.xlabel("Confidence")
plt.ylabel("Number of Coins")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nNext Steps:")
print("- Improve directional accuracy beyond 60%")
print("- Validate performance across more assets and timeframes")
print("- Incorporate market volume/liquidity to assess trade viability")
print("- Track live predictions and trade execution capability")