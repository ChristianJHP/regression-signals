import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from regression import add_technical_indicators

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root directory to Python path
sys.path.append(project_root)

def load_models():
    """Load all saved models from the models directory."""
    models = {}
    models_dir = os.path.join(project_root, 'models')
    
    if not os.path.exists(models_dir):
        print("Models directory not found. Please train models first.")
        return None
        
    for filename in os.listdir(models_dir):
        if filename.endswith('_classifier.joblib'):
            symbol = filename.replace('_classifier.joblib', '')
            model_path = os.path.join(models_dir, filename)
            try:
                models[symbol] = joblib.load(model_path)
                print(f"Loaded model for {symbol}")
            except Exception as e:
                print(f"Error loading model for {symbol}: {str(e)}")
    
    if not models:
        print("No models found. Please train models first.")
        return None
        
    return models

def fetch_latest_data(symbols, days=30, future_days=1):
    """Fetch latest data for given symbols and prepare for future prediction."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    future_end_date = end_date + timedelta(days=future_days)
    
    all_data = []
    failed_symbols = []
    
    for symbol in symbols:
        try:
            # Add -USD suffix for Yahoo Finance
            ticker = f"{symbol}-USD"
            # Fetch historical data
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                print(f"No data found for {symbol}")
                failed_symbols.append(symbol)
                continue
                
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Create future dates
            future_dates = pd.date_range(start=end_date + timedelta(days=1), 
                                       end=future_end_date, 
                                       freq='D')
            
            # Create future data frame with the same structure
            future_data = pd.DataFrame({
                'Date': future_dates,
                'Open': data['Open'].iloc[-1],  # Use last known values
                'High': data['High'].iloc[-1],
                'Low': data['Low'].iloc[-1],
                'Close': data['Close'].iloc[-1],
                'Volume': data['Volume'].iloc[-1]
            })
            
            # Combine historical and future data
            data = pd.concat([data, future_data], ignore_index=True)
            
            # Rename columns to match our format
            data = data.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add symbol column
            data['symbol'] = symbol
            
            all_data.append(data)
            print(f"Successfully fetched data for {symbol}")
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            failed_symbols.append(symbol)
    
    if not all_data:
        print("No data could be fetched for any symbol.")
        return None, failed_symbols
        
    # Combine all data
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Verify all required columns are present
    required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in full_df.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None, failed_symbols
        
    return full_df, failed_symbols

def make_predictions(models, data):
    """Make predictions using the loaded models."""
    if data is None or data.empty:
        return None
        
    # Add technical indicators
    data = add_technical_indicators(data)
    
    predictions = {}
    
    for symbol, model in models.items():
        # Get data for this symbol
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            print(f"No data available for {symbol}")
            continue
            
        # Prepare features
        features = ['log_return', 'macd_hist', 'daily_range_pct',
                   'volume_change', 'price_to_support', 'close_lag_1',
                   'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d']
        
        # Check if all features are present
        missing_features = [f for f in features if f not in symbol_data.columns]
        if missing_features:
            print(f"Missing features for {symbol}: {missing_features}")
            continue
            
        # Make predictions
        try:
            X = symbol_data[features]
            predictions[symbol] = model.predict_proba(X)[:, 1]  # Probability of class 1
            print(f"Made predictions for {symbol}")
        except Exception as e:
            print(f"Error making predictions for {symbol}: {str(e)}")
    
    return predictions

def save_predictions(predictions, data):
    """Save predictions to CSV files."""
    if predictions is None or not predictions:
        return
        
    signals_dir = os.path.join(project_root, 'data', 'signals')
    os.makedirs(signals_dir, exist_ok=True)
    
    current_date = datetime.now()
    
    for symbol, probs in predictions.items():
        # Get the corresponding data
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            continue
            
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'timestamp': symbol_data['timestamp'],
            'symbol': symbol,
            'close': symbol_data['close'],
            'pump_probability': probs,
            'is_future_prediction': symbol_data['timestamp'] > current_date
        })
        
        # Sort by timestamp
        pred_df = pred_df.sort_values('timestamp')
        
        # Save to CSV
        output_file = os.path.join(signals_dir, f'{symbol}_predictions.csv')
        pred_df.to_csv(output_file, index=False)
        print(f"Saved predictions for {symbol} to {output_file}")
        
        # Print future predictions
        future_preds = pred_df[pred_df['is_future_prediction']]
        if not future_preds.empty:
            print(f"\nFuture predictions for {symbol}:")
            for _, row in future_preds.iterrows():
                print(f"Date: {row['timestamp'].strftime('%Y-%m-%d')}, "
                      f"Pump Probability: {row['pump_probability']:.2%}")

def main():
    # Load models
    models = load_models()
    if not models:
        return
        
    print(f"Loaded models for symbols: {list(models.keys())}")
    
    # Fetch latest data including tomorrow's prediction
    data, failed_symbols = fetch_latest_data(list(models.keys()), days=30, future_days=1)
    if data is None:
        print("No valid data available to make predictions, exiting.")
        return
        
    # Make predictions
    predictions = make_predictions(models, data)
    if predictions is None:
        print("No predictions could be made, exiting.")
        return
        
    # Save predictions
    save_predictions(predictions, data)
    
    if failed_symbols:
        print(f"\nFailed to fetch data for: {failed_symbols}")
    
    print("\nPrediction process completed.")

if __name__ == "__main__":
    main()