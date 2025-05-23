import os
import pandas as pd
import numpy as np
import joblib

# Set target prediction date
target_date = pd.Timestamp("2025-05-22")

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(project_root, "models")
price_path = os.path.join(project_root, "data", "crypto_ohlcv.csv")

# Load price data
df = pd.read_csv(price_path, parse_dates=["timestamp"])
df["symbol"] = df["symbol"].str.upper()
symbols = df["symbol"].unique()

# Results
predictions = []

for symbol in symbols:
    print(f"Processing {symbol}...")
    
    model_path = os.path.join(model_dir, f"{symbol}_classifier.joblib")
    if not os.path.exists(model_path):
        print(f"Classifier for {symbol} not found. Skipping.")
        continue
        
    try:
        model = joblib.load(model_path)
        df_symbol = df[df["symbol"] == symbol].copy()
        df_symbol.set_index("timestamp", inplace=True)
        df_symbol.sort_index(inplace=True)

        # Generate required features
        df_symbol["close_lag_1"] = df_symbol["close"].shift(1)
        df_symbol["close_lag_2"] = df_symbol["close"].shift(2)
        df_symbol["volume_lag_1"] = df_symbol["volume"].shift(1)
        df_symbol["log_return"] = (df_symbol["close"] / df_symbol["close"].shift(1)).apply(lambda x: np.log(x) if x > 0 else 0)
        df_symbol["rolling_volatility_7d"] = df_symbol["log_return"].rolling(window=42).std()
        df_symbol["ema_20"] = df_symbol["close"].ewm(span=20, adjust=False).mean()
        df_symbol["macd_hist"] = (
            df_symbol["close"].ewm(span=12, adjust=False).mean() -
            df_symbol["close"].ewm(span=26, adjust=False).mean()
        )
        df_symbol["daily_range_pct"] = (df_symbol["high"] - df_symbol["low"]) / df_symbol["low"]
        df_symbol["price_to_support"] = df_symbol["close"] / df_symbol["low"].rolling(window=14).min()

        # Join ETH log return
        eth = df[df["symbol"] == "ETH"].copy()
        eth.set_index("timestamp", inplace=True)
        eth["log_return"] = (eth["close"] / eth["close"].shift(1)).apply(lambda x: np.log(x) if x > 0 else 0)
        eth["eth_log_return_lag1"] = eth["log_return"].shift(1)

        df_symbol = df_symbol.join(eth[["eth_log_return_lag1"]], how="left")

        # Get the most recent data point and use it for May 22nd prediction
        latest_data = df_symbol.iloc[-1:].copy()
        if latest_data.empty:
            print(f"No data available for {symbol}")
            continue

        feature_cols = [
            "close_lag_1", "close_lag_2", "volume_lag_1", "rolling_volatility_7d",
            "eth_log_return_lag1", "ema_20", "macd_hist", "daily_range_pct", "price_to_support"
        ]

        if not all(col in latest_data.columns for col in feature_cols):
            print(f"Missing one or more required features for {symbol}. Skipping.")
            continue

        X = latest_data[feature_cols].values.reshape(1, -1)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]  # Probability of class 1
        label = "PUMP" if pred == 1 else "NO PUMP"

        print(f"{symbol}: {label} (Confidence: {prob:.2%})")
        predictions.append((symbol, label, prob))

    except Exception as e:
        print(f"Error generating prediction for {symbol}: {e}")

print("\nFinal predictions for May 22nd:")
for sym, pred, prob in sorted(predictions, key=lambda x: x[2], reverse=True):  # Sort by confidence
    print(f"{sym}: {pred} (Confidence: {prob:.2%})")
