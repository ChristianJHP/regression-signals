

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Define the coins and their predictions
predictions = {
    "ARB": 0.45570550949879984,
    "RUNE": 2.1643848850963185,
    "SOL": 183.9429032261923,
    "ADA": 0.7931736097473553,
    "AVAX": 23.86265472619042,
    "ETH": 2592.1755531817216,
    "RNDR": 4.730516427128372,
    "INJ": 12.282520598042302,
    "FET": 0.797756525017283,
    "MATIC": 0.21780850924520173,
}

# Map to Yahoo Finance symbols (or other reliable identifiers)
symbol_map = {
    "ARB": "ARB-USD",
    "RUNE": "RUNE-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "ETH": "ETH-USD",
    "RNDR": "RNDR-USD",
    "INJ": "INJ-USD",
    "FET": "FET-USD",
    "MATIC": "MATIC-USD",
}

# Fetch current prices
actual_prices = {}
for coin, yf_symbol in symbol_map.items():
    try:
        ticker = yf.Ticker(yf_symbol)
        price = ticker.history(period="1d")["Close"].iloc[-1]
        actual_prices[coin] = price
    except Exception as e:
        actual_prices[coin] = None
        print(f"Failed to fetch {coin}: {e}")

# Create dataframe
df = pd.DataFrame([
    {"symbol": coin, "predicted": pred, "actual": actual_prices.get(coin)}
    for coin, pred in predictions.items()
])

df["diff"] = df["actual"] - df["predicted"]
df["pct_diff"] = 100 * df["diff"] / df["predicted"]

# Plot
plt.figure(figsize=(12, 6))
plt.bar(df["symbol"], df["pct_diff"], color="skyblue")
plt.axhline(0, color="black", linestyle="--")
plt.title("Percentage Difference: Predicted vs Actual Prices (May 22, 2025)")
plt.ylabel("% Difference")
plt.xlabel("Coin")
plt.grid(True)
plt.tight_layout()
plt.show()