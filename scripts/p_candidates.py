import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Define directories
PUMP_CSV_PATH = "data/pump_candidates_20250523_0426.csv"
OUTPUT_DIR = "data/pump_ohlcv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

successful_downloads = []

# Load trending coins
df = pd.read_csv(PUMP_CSV_PATH)
symbols = df["symbol"].unique()

def to_ticker_format(name):
    name = name.upper()
    name = name.replace("(", "").replace(")", "").replace(".", "").replace(",", "")
    name = name.replace(" ", "")
    name = name.split()[0]  # fallback: take first token
    return name + "-USD"

# Fetch OHLCV data
for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    try:
        ticker = to_ticker_format(symbol)
        coin_data = yf.download(ticker, start="2024-01-01", end=datetime.today().strftime('%Y-%m-%d'), interval="1d")
        if coin_data.empty:
            print(f"No data for {symbol}")
            continue
        coin_data.to_csv(os.path.join(OUTPUT_DIR, f"{symbol}.csv"))
        successful_downloads.append(symbol)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

# Step 2: Flag suspicious coins (basic heuristic detection)
def flag_suspicious_behavior(coin_df):
    suspicious = False
    try:
        recent = coin_df.tail(8)  # Last 7 days + today
        close_prices = pd.to_numeric(recent['Close'], errors='coerce')
        volume = pd.to_numeric(recent['Volume'], errors='coerce')

        if close_prices.isna().any() or volume.isna().any():
            return False

        pct_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        avg_vol = volume[:-1].mean()
        vol_spike = volume.iloc[-1] > 5 * avg_vol
        flat_after_surge = close_prices.diff().abs().iloc[-3:].sum() < 0.02 * close_prices.mean()

        if pct_change > 1.0 or vol_spike or flat_after_surge:
            suspicious = True
    except Exception as e:
        print(f"Error in heuristic for {symbol}: {e}")
    return suspicious

# Scan each saved file for suspicious behavior
flagged = []
for symbol in symbols:
    filepath = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    if os.path.exists(filepath):
        df_coin = pd.read_csv(filepath)
        is_pump = flag_suspicious_behavior(df_coin)
        df_coin['label'] = 1 if is_pump else 0
        df_coin.to_csv(filepath, index=False)
        if is_pump:
            flagged.append(symbol)

print("\nPotential pump candidates based on heuristics:")
print(flagged)

print("\nSuccessfully downloaded and processed:")
print(successful_downloads)
