import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Load trained models
try:
    models = joblib.load("crypto_models.joblib")
    print("Successfully loaded models for symbols:", list(models.keys()))
except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit(1)

# Load historical data
df = pd.read_csv("crypto_ohlcv.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[df['timestamp'] <= "2025-05-21"]
df = df.sort_values(['symbol', 'timestamp'])

# Store predictions
predictions = []

for symbol in models.keys():
    print(f"\nProcessing {symbol}...")
    symbol_data = df[df['symbol'] == symbol].copy()
    if len(symbol_data) < 30:
        print(f"Not enough data for {symbol}, skipping...")
        continue

    # Compute features
    def calc_ema(data, span):
        alpha = 2 / (span + 1)
        ema = [data.iloc[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data.iloc[i] + (1 - alpha) * ema[i-1])
        return pd.Series(ema, index=data.index)

    symbol_data['log_return'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
    symbol_data['ema_20'] = calc_ema(symbol_data['close'], 20).shift(1)
    exp1 = symbol_data['close'].ewm(span=12, adjust=False).mean().shift(1)
    exp2 = symbol_data['close'].ewm(span=26, adjust=False).mean().shift(1)
    symbol_data['macd'] = exp1 - exp2
    symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, adjust=False).mean().shift(1)
    symbol_data['macd_hist'] = symbol_data['macd'] - symbol_data['macd_signal']
    symbol_data['daily_range_pct'] = (symbol_data['high'].shift(1) - symbol_data['low'].shift(1)) / symbol_data['close'].shift(1)
    symbol_data['volume_change'] = symbol_data['volume'].pct_change().shift(1)
    symbol_data['support_level'] = symbol_data['low'].rolling(20).min().shift(1)
    symbol_data['price_to_support'] = symbol_data['close'].shift(1) / symbol_data['support_level']
    symbol_data.dropna(inplace=True)

    if symbol_data.empty:
        print(f"No valid data after feature calculation for {symbol}, skipping...")
        continue

    try:
        # Get the last row of features
        feature_columns = models[symbol]['feature_columns']
        X = symbol_data[feature_columns].iloc[-1:]
        
        # Scale features using the saved scaler
        scaler = models[symbol]['scaler']
        X_scaled = scaler.transform(X)
        
        # Make prediction
        model = models[symbol]['model']
        predicted = model.predict(X_scaled)[0]
        latest_price = symbol_data['close'].iloc[-1]
        last_timestamp = symbol_data['timestamp'].iloc[-1]

        predictions.append({
            'symbol': symbol,
            'as_of': last_timestamp,
            'predicted_price_for_may22': predicted,
            'actual_price_may21': latest_price,
            'predicted_change_pct': 100 * (predicted - latest_price) / latest_price,
            'model_type': models[symbol]['model_type']
        })
        print(f"Successfully generated prediction for {symbol}")
    except Exception as e:
        print(f"Error generating prediction for {symbol}: {str(e)}")
        continue

if not predictions:
    print("No predictions were generated. Check the data and models.")
    exit(1)

# Output results
pred_df = pd.DataFrame(predictions).sort_values('predicted_change_pct', ascending=False)
pred_df.to_csv("graphs/predicted_may22_prices.csv", index=False)

print("\nPredictions for May 22nd:")
print("=" * 80)
for _, row in pred_df.iterrows():
    print(f"\n{row['symbol']}:")
    print(f"Current Price (May 21): ${row['actual_price_may21']:.2f}")
    print(f"Predicted Price (May 22): ${row['predicted_price_for_may22']:.2f}")
    print(f"Predicted Change: {row['predicted_change_pct']:.2f}%")
    print(f"Model Type: {row['model_type']}")
print("=" * 80)
print("\nPredictions saved to 'graphs/predicted_may22_prices.csv'")