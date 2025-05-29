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
    """Load and clean data from CSV, removing duplicate headers and empty rows."""
    print("Loading main OHLCV dataset...")
    
    # Load raw data as plain text
    df = pd.read_csv(csv_path, header=None)
    
    # Drop completely empty rows
    df = df.dropna(how="all")
    
    # Remove any rows where the first column is 'timestamp' (duplicate header rows)
    df = df[df.iloc[:, 0] != "timestamp"]
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    # Assign expected columns (adjust if you have more!)
    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    if len(df.columns) >= len(expected_columns):
        df.columns = expected_columns + [f'extra_{i}' for i in range(len(df.columns) - len(expected_columns))]
    else:
        print("WARNING: Not enough columns to assign all headers. Check data manually!")
    
    # Drop extra columns if they exist
    df = df[expected_columns]
    
    # Convert numeric columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    print(df.dtypes)
    print(df.head())
    print(df.tail())
    
    return df

def main():
    # Load data and clean
    data_dir = os.path.join(project_root, 'data')
    price_path = os.path.join(data_dir, 'crypto_ohlcv.csv')
    full_df = load_and_prepare_data(price_path)
    
    # Add technical indicators
    full_df = add_technical_indicators(full_df)
    
    # Compute rolling z-scores
    full_df['return_1d'] = full_df['close'].pct_change()
    full_df['z_score_return'] = (full_df['return_1d'] - full_df['return_1d'].rolling(window=14).mean()) / full_df['return_1d'].rolling(window=14).std()
    full_df['z_score_volume'] = (full_df['volume'] - full_df['volume'].rolling(window=14).mean()) / full_df['volume'].rolling(window=14).std()

    return_thresh = full_df.groupby('symbol')['z_score_return'].transform(lambda x: x.dropna().quantile(0.65))
    volume_thresh = full_df.groupby('symbol')['z_score_volume'].transform(lambda x: x.dropna().quantile(0.65))

    # Label the day before a pump occurs
    pump_label = ((full_df['z_score_return'] > return_thresh) & (full_df['z_score_volume'] > volume_thresh)).astype(int)
    full_df['target'] = pump_label.shift(-1).fillna(0).astype(int)
    print("Pump label distribution:\n", full_df['target'].value_counts())

    # Drop unnecessary columns and rows with NaNs
    full_df = full_df.drop(columns=['ema_20'], errors='ignore').dropna()
    if full_df.empty:
        print("No usable data after cleaning. Exiting.")
        return

    # Train separate model for each coin
    features = ['log_return', 'macd_hist', 'daily_range_pct',
                 'volume_change', 'price_to_support', 'close_lag_1',
                 'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d']

    for coin_symbol in full_df['symbol'].unique():
        coin_df = full_df[full_df['symbol'] == coin_symbol].copy()
        if coin_df['target'].sum() == 0:
            print(f"Skipping {coin_symbol}: No positive samples.")
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

        # Save high-confidence predictions
        probs = model.predict_proba(X)[:, 1]
        coin_df['confidence'] = probs
        coin_df['prediction'] = (probs > 0.6).astype(int)

        high_confidence = coin_df[coin_df['prediction'] == 1].copy()
        if not high_confidence.empty:
            signal_path = os.path.join(project_root, 'data', f'{coin_symbol}_signals.csv')
            high_confidence[['timestamp', 'symbol', 'confidence', 'prediction', 'target']].to_csv(signal_path, index=False)
            print(f"Saved high-confidence predictions for {coin_symbol} to {signal_path}")
        else:
            print(f"No high-confidence predictions for {coin_symbol}")

        # Naive baseline: predict using only close_lag_1
        close_lag_1_idx = X.columns.get_loc("close_lag_1")
        from sklearn.linear_model import LinearRegression
        naive_model = LinearRegression()
        # Drop rows with NaNs in X_train[:, [close_lag_1_idx]] or y_train
        naive_X = X_train.iloc[:, [close_lag_1_idx]].copy().to_numpy()
        naive_y = y_train.copy().to_numpy()

        # Remove rows with NaNs in either
        mask = ~np.isnan(naive_X).flatten() & ~np.isnan(naive_y)
        naive_X = naive_X[mask]
        naive_y = naive_y[mask]

        # Only fit if there's enough data
        if len(naive_X) > 0:
            naive_model.fit(naive_X, naive_y)
            naive_r2 = naive_model.score(X_test.iloc[:, [close_lag_1_idx]].to_numpy(), y_test.to_numpy())
            print(f"Naive Baseline R2 (Close t-1 only): {naive_r2:.4f}")
        else:
            print(f"Skipping naive baseline for {coin_symbol} due to no valid data.")

if __name__ == "__main__":
    main()