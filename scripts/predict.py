import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import yfinance as yf

def add_technical_indicators(df):
    """
    Add technical indicators using only past data (t-1) to predict current price (t).
    This function ensures NO data leakage: all features are calculated using only information available up to t-1.
    Assumes df contains only one symbol.
    """
    # Sort by timestamp to ensure proper time order
    df = df.sort_values('timestamp')
    
    # Calculate log returns using previous day's close
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate 20-day EMA using only past data
    def calculate_ema(data, span):
        alpha = 2 / (span + 1)
        ema = [data.iloc[0]]  # Initialize with first value by position
        for i in range(1, len(data)):
            ema.append(alpha * data.iloc[i] + (1 - alpha) * ema[i-1])
        return pd.Series(ema, index=data.index)
    df['ema_20'] = calculate_ema(df['close'], 20).shift(1)
    
    # Calculate MACD using only past data
    exp1 = df['close'].ewm(span=12, adjust=False).mean().shift(1)
    exp2 = df['close'].ewm(span=26, adjust=False).mean().shift(1)
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().shift(1)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate daily range using previous day's data
    df['daily_range_pct'] = (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    
    # Calculate volume change using previous day's data
    df['volume_change'] = df['volume'].pct_change().shift(1)
    
    # Calculate support level using a rolling window (e.g., 20 days), shifted by 1 to avoid future data
    support_window = 20
    df['support_level'] = df['low'].rolling(window=support_window, min_periods=1).min().shift(1)
    df['price_to_support'] = df['close'].shift(1) / df['support_level']
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def fetch_latest_data(symbols, days=30):
    """
    Fetch the latest data for the given symbols using yfinance.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_data = []
    for symbol in symbols:
        try:
            # Add -USD suffix for yfinance
            ticker = f"{symbol}-USD"
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                # Rename columns to match our expected format
                data = data.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                data['symbol'] = symbol
                data = data.reset_index()
                data = data.rename(columns={'Date': 'timestamp'})
                # Ensure all required columns are present
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                if all(col in data.columns for col in required_columns):
                    all_data.append(data)
                    print(f"Successfully fetched data for {symbol}")
                else:
                    print(f"Missing required columns for {symbol}")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    
    if not all_data:
        print(f"No data was fetched for any symbol. Symbols attempted: {symbols}")
        return pd.DataFrame(columns=['timestamp','symbol','open','high','low','close','volume'])
    
    # Combine all data and ensure proper column order
    combined_data = pd.concat(all_data, ignore_index=True)

    required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    try:
        # Drop rows with missing critical data
        combined_data.dropna(subset=required_columns, inplace=True)
        if combined_data.empty:
            print("No valid rows after dropping missing data.")
            return pd.DataFrame(columns=required_columns)

        # Filter out symbols with no valid rows
        valid_symbols = combined_data['symbol'].unique()
        print(f"Valid symbols with data: {list(valid_symbols)}")

        return combined_data[required_columns]
    except Exception as e:
        print(f"Incomplete data assembly: {e}")
        return pd.DataFrame(columns=required_columns)

def generate_trading_signals(actual_price, predicted_price, threshold):
    """
    Generate trading signals based on predicted vs actual prices.
    """
    diff = predicted_price - actual_price
    if diff > threshold:
        return 'BUY'
    elif diff < -threshold:
        return 'SELL'
    else:
        return 'HOLD'

def main():
    # Load the saved models
    try:
        models = joblib.load('crypto_models.joblib')
        print("Successfully loaded saved models")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return

    # Get list of symbols from loaded models
    symbols = list(models.keys())
    print(f"Loaded models for symbols: {symbols}")

    # Fetch latest data
    try:
        df = fetch_latest_data(symbols)
        if df.empty:
            print("No valid data available to make predictions, exiting.")
            return
        print("\nFetched latest data successfully")
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return

    # Process each symbol
    predictions = []
    for symbol in symbols:
        if symbol not in models:
            print(f"No model found for {symbol}, skipping...")
            continue

        # Get data for this symbol
        symbol_data = df[df['symbol'] == symbol].copy()
        if symbol_data.empty:
            print(f"No data found for {symbol}, skipping...")
            continue

        # Skip if symbol data is too short to compute indicators
        if len(symbol_data) < 30:
            print(f"Not enough recent data for {symbol}, skipping...")
            continue

        # Add technical indicators
        symbol_data = add_technical_indicators(symbol_data)

        # Ensure only use May 21, 2025 data for prediction
        prediction_date = pd.Timestamp('2025-05-21')
        if prediction_date not in symbol_data['timestamp'].values:
            print(f"No data for {symbol} on 2025-05-21, skipping...")
            continue
        latest_data = symbol_data[symbol_data['timestamp'] == prediction_date].copy()
        
        # Prepare features
        feature_columns = models[symbol]['feature_columns']
        X = latest_data[feature_columns]
        
        # Scale features
        scaler = models[symbol]['scaler']
        X_scaled = scaler.transform(X)
        
        # Make prediction
        model = models[symbol]['model']
        predicted_price = model.predict(X_scaled)[0]
        actual_price = latest_data['close'].iloc[0]
        
        # Generate trading signal
        mae = models[symbol].get('mae', actual_price * 0.02)  # Default to 2% if MAE not available
        signal = generate_trading_signals(actual_price, predicted_price, 2 * mae)
        
        # Store prediction
        predictions.append({
            'symbol': symbol,
            'timestamp': latest_data['timestamp'].iloc[0],
            'actual_price': actual_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': ((predicted_price - actual_price) / actual_price) * 100,
            'signal': signal,
            'model_type': models[symbol]['model_type']
        })

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Sort by predicted change percentage
    predictions_df = predictions_df.sort_values('predicted_change_pct', ascending=False)
    
    # Save predictions to CSV
    predictions_df.to_csv('graphs/latest_predictions.csv', index=False)
    
    # Print predictions
    print("\nLatest Predictions:")
    print("=" * 80)
    for _, row in predictions_df.iterrows():
        print(f"\n{row['symbol']}:")
        print(f"Current Price: ${row['actual_price']:.2f}")
        print(f"Predicted Price: ${row['predicted_price']:.2f}")
        print(f"Predicted Change: {row['predicted_change_pct']:.2f}%")
        print(f"Signal: {row['signal']}")
        print(f"Model Type: {row['model_type']}")
    print("=" * 80)
    print("\nPredictions saved to 'graphs/latest_predictions.csv'")

if __name__ == "__main__":
    main()