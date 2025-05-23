import os
import sys
import pandas as pd

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the data
df = pd.read_csv(os.path.join(project_root, "data", "crypto_ohlcv.csv"))
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter data up to May 22nd
df = df[df['timestamp'] <= "2025-05-22"]

# Get unique symbols
symbols = df['symbol'].unique()

print("\nLast 5 days of prices for each coin:")
print("=" * 80)

for symbol in symbols:
    # Get data for this symbol
    symbol_data = df[df['symbol'] == symbol].copy()
    
    # Sort by timestamp and get last 5 days
    symbol_data = symbol_data.sort_values('timestamp')
    last_5_days = symbol_data.tail(5)
    
    print(f"\n{symbol}:")
    print("-" * 40)
    for _, row in last_5_days.iterrows():
        print(f"{row['timestamp'].strftime('%Y-%m-%d')}: ${row['close']:.2f}")
    print("-" * 40)

print("\nData saved to 'graphs/last_5_days_prices.csv'")

# Save to CSV
last_5_days_all = pd.concat([df[df['symbol'] == symbol].sort_values('timestamp').tail(5) for symbol in symbols])
last_5_days_all.to_csv(os.path.join(project_root, "graphs", "last_5_days_prices.csv"), index=False) 