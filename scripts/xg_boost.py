import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Now import from regression module
from regression import add_technical_indicators

def load_and_prepare_data(csv_path):
    """Load and prepare data from a CSV file with error handling."""
    try:
        # Read CSV and handle flexible date/timestamp columns
        df = pd.read_csv(csv_path)
        if 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("CSV must contain either 'Date' or 'timestamp' column.")
        # Rename columns to match expected format (handled dynamically for timestamp)
        df = df.rename(columns={
            # 'Date': 'timestamp',  # handled above
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        # Extract symbol from filename, but handle combined file
        symbol = os.path.basename(csv_path).replace('.csv', '')
        if symbol == "crypto_ohlcv" and 'symbol' in df.columns:
            # Use the symbol column from the file itself
            pass  # symbols already present
        else:
            df['symbol'] = symbol
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {str(e)}")
        return None

def main():
    # Load all CSV files
    data_dir = os.path.join(project_root, 'data')
    all_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            csv_path = os.path.join(data_dir, filename)
            df = load_and_prepare_data(csv_path)
            # Check for required OHLCV columns and at least 10 non-null close values
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            if (
                df is not None and
                required_cols.issubset(df.columns) and
                df['close'].notnull().sum() > 10
            ):
                all_data.append(df)
            else:
                print(f"Skipping {filename}: not enough usable rows.")
    
    if not all_data:
        print("No valid data files found. Exiting.")
        return
        
    # Combine all data
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Add technical indicators
    full_df = add_technical_indicators(full_df)
    
    # Compute rolling z-scores
    full_df['return_1d'] = full_df['close'].pct_change()
    full_df['z_score_return'] = (full_df['return_1d'] - full_df['return_1d'].rolling(window=14).mean()) / full_df['return_1d'].rolling(window=14).std()
    full_df['z_score_volume'] = (full_df['volume'] - full_df['volume'].rolling(window=14).mean()) / full_df['volume'].rolling(window=14).std()

    return_thresh = full_df.groupby('symbol')['z_score_return'].transform(lambda x: x.dropna().quantile(0.65))
    volume_thresh = full_df.groupby('symbol')['z_score_volume'].transform(lambda x: x.dropna().quantile(0.65))
    # Label the day *before* a pump occurs to enable early prediction
    pump_label = ((full_df['z_score_return'] > return_thresh) & (full_df['z_score_volume'] > volume_thresh)).astype(int)
    full_df['target'] = pump_label.shift(-1).fillna(0).astype(int)
    print("Pump label distribution:\n", full_df['target'].value_counts())
    
    # Store ema_20 for debugging if present
    ema_20_debug = full_df['ema_20'].copy() if 'ema_20' in full_df.columns else None

    # Drop 'ema_20' column if present
    full_df = full_df.drop(columns=['ema_20'], errors='ignore')
    
    # Drop rows with NaN values
    full_df = full_df.dropna()
    if full_df.empty:
        print("No usable data after cleaning. Exiting.")
        return
    
    # Split features and target
    features = ['log_return', 'macd_hist', 'daily_range_pct',
                'volume_change', 'price_to_support', 'close_lag_1',
                'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d']

    # Train a separate model for each coin
    for coin_symbol in full_df['symbol'].unique():
        coin_df = full_df[full_df['symbol'] == coin_symbol].copy()
        if coin_df['target'].sum() == 0:
            print(f"Skipping {coin_symbol}: No positive samples.")
            continue
        coin_df = coin_df.drop(columns=['ema_20'], errors='ignore').dropna()
        if coin_df.empty:
            print(f"Skipping {coin_symbol}: No usable data after cleaning.")
            continue

        X = coin_df[features]
        y = coin_df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale = neg / pos if pos > 0 else 1

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        model_path = os.path.join(project_root, 'models', f'{coin_symbol}_classifier.joblib')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Saved model for {coin_symbol} to {model_path}")

        # Predict on all available data
        probs = model.predict_proba(X)[:, 1]  # Confidence scores for positive class
        coin_df['confidence'] = probs
        coin_df['prediction'] = (probs > 0.6).astype(int)

        # Filter high-confidence positive predictions
        high_confidence = coin_df[coin_df['prediction'] == 1].copy()
        if not high_confidence.empty:
            signal_path = os.path.join(project_root, 'data', f'{coin_symbol}_signals.csv')
            high_confidence[['timestamp', 'symbol', 'confidence', 'prediction', 'target']].to_csv(signal_path, index=False)
            print(f"Saved high-confidence predictions for {coin_symbol} to {signal_path}")
        else:
            print(f"No high-confidence predictions for {coin_symbol}")

    import matplotlib.pyplot as plt

    # Visualize predictions
    for coin_symbol in full_df['symbol'].unique():
        signal_path = os.path.join(data_dir, f"{coin_symbol}_signals.csv")
        if not os.path.exists(signal_path):
            continue
        df_signals = pd.read_csv(signal_path, parse_dates=['timestamp'])
        df_coin = full_df[full_df['symbol'] == coin_symbol].copy()
        df_coin = df_coin.set_index('timestamp').sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_coin.index, df_coin['close'], label='Close Price')
        for ts in df_signals['timestamp']:
            if ts in df_coin.index:
                plt.axvline(x=ts, color='red', linestyle='--', alpha=0.6)

        plt.title(f"{coin_symbol} Close Price with Predicted Pumps")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(data_dir, f"{coin_symbol}_pump_predictions.png"))
        plt.close()

if __name__ == "__main__":
    main()