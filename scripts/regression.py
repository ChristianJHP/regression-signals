import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def add_technical_indicators(df):
    """
    Add technical indicators using only past data (t-1) to predict current price (t).
    This function ensures NO data leakage: all features are calculated using only information available up to t-1.
    """
    # Sort by timestamp to ensure proper time order
    df = df.sort_values('timestamp')
    
    # Calculate log returns using previous day's close
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate 20-day EMA using only past data (calculate, then shift)
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean().shift(1))

    # Naive baseline: previous day's close
    df['naive_baseline'] = df['close'].shift(1)

    # Calculate MACD using only past data (calculate, then shift)
    exp1 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean().shift(1))
    exp2 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean().shift(1))
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean().shift(1))
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Calculate daily range using previous day's data
    df['daily_range_pct'] = (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)

    # Calculate volume change using previous day's data
    df['volume_change'] = df.groupby('symbol')['volume'].transform(lambda x: x.pct_change().shift(1))

    # Calculate support level using a rolling window (e.g., 20 days), shifted by 1 to avoid future data
    df['support_level'] = df.groupby('symbol')['low'].transform(lambda x: x.rolling(window=20, min_periods=1).min().shift(1))
    df['price_to_support'] = df['close'].shift(1) / df['support_level']

    # Lag features
    df['close_lag_1'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(1))
    df['close_lag_2'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(2))
    df['volume_lag_1'] = df.groupby('symbol')['volume'].transform(lambda x: x.shift(1))

    # Rolling volatility (7-day), shifted by 1 to avoid leakage
    df['rolling_volatility_7d'] = df.groupby('symbol')['log_return'].transform(lambda x: x.rolling(7, min_periods=1).std().shift(1))

    # Cross-asset feature (e.g., ETH log return as signal for other coins)
    eth_returns = df[df['symbol'] == 'ETH'][['timestamp', 'log_return']].copy()
    eth_returns.rename(columns={'log_return': 'eth_log_return_lag1'}, inplace=True)
    eth_returns['eth_log_return_lag1'] = eth_returns['eth_log_return_lag1'].shift(1)
    df = df.merge(eth_returns, on='timestamp', how='left')
    df['eth_log_return_lag1'] = df['eth_log_return_lag1'].fillna(0)

    # Print debug lines for feature inspection
    print("Feature check:")
    print(df[['timestamp', 'symbol', 'close', 'log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support']].tail(5))

    # Drop the first rows where we don't have t-1 data for all features
    print("Checking for NaNs before drop:")
    print(df[['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support',
              'close_lag_1', 'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d', 'eth_log_return_lag1']].isna().sum())
    print(f"Before dropna: {len(df)} rows")
    
    # Only drop rows where we don't have enough data for prediction
    df = df.dropna(subset=[
        'close_lag_1',  # We need at least one previous price
        'ema_20',       # We need the EMA
        'macd_hist'     # We need the MACD
    ])
    print(f"After dropna: {len(df)} rows")

    # Fill remaining NaN values with 0 for features that can be 0
    fill_zero_cols = ['volume_change', 'rolling_volatility_7d', 'eth_log_return_lag1']
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    # Fill remaining NaN values with forward fill for other features
    df = df.ffill()

    # Validate that features do not correlate with current day's close
    print("\nValidating feature timing (correlation with current day's close):")
    print("=" * 80)
    for feature in ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support']:
        try:
            corr = df[feature].corr(df['close'])
            if np.isnan(corr):
                print(f"OK: {feature} correlation is NaN (likely due to insufficient data)")
            elif abs(corr) > 0.5:
                print(f"WARNING: {feature} shows high correlation ({corr:.3f}) with current day's close price (possible leakage)")
            else:
                print(f"OK: {feature} shows low correlation ({corr:.3f}) with current day's close price")
        except Exception as e:
            print(f"Error calculating correlation for {feature}: {e}")
    print("=" * 80)

    return df

# --- Trading Signal Generation ---
def generate_trading_signals(y_test, y_pred, threshold):
    """
    Generate trading signals ('BUY', 'SELL', 'HOLD') based on predicted vs actual prices.
    A 'BUY' is signaled if predicted > actual + threshold,
    a 'SELL' if predicted < actual - threshold,
    otherwise 'HOLD'.
    Also returns the signal strength (predicted change pct).
    """
    signals = []
    signal_strengths = []
    for actual, predicted in zip(y_test, y_pred):
        diff = predicted - actual
        pct_change = (predicted - actual) / actual * 100 if actual != 0 else 0
        signal_strengths.append(pct_change)
        if diff > threshold:
            signals.append('BUY')
        elif diff < -threshold:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals, signal_strengths

# --- Backtest Trading Signals ---
def backtest_signals(prices, signals):
    """
    Backtest a simple strategy based on predicted trading signals.
    Assumes execution at today's close, and sells at the next day's close.
    Returns list of daily profits and a cumulative return.
    """
    profits = []
    for i in range(len(signals) - 1):  # we can't trade on the last day
        if signals[i] == 'BUY':
            profits.append(prices[i + 1] - prices[i])  # buy and sell next day
        elif signals[i] == 'SELL':
            profits.append(prices[i] - prices[i + 1])  # short sell
        else:
            profits.append(0)
    profits.append(0)  # last day, no trade
    return profits

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)"""
    # Calculate True Range
    df['tr'] = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Calculate smoothed averages
    df['tr_smoothed'] = df['tr'].rolling(window=period).sum()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).sum() / df['tr_smoothed'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).sum() / df['tr_smoothed'])
    
    # Calculate ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    return df['adx']

# Load your data
df = pd.read_csv(os.path.join(project_root, "data", "crypto_ohlcv.csv"))
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter out BTC and add more altcoins
coins = ['ETH', 'BNB', 'SOL', 'AVAX', 'MATIC', 'ADA', 'INJ', 'RUNE', 'FET', 'RNDR', 'KAS', 'TWT', 'ARB', 'BPX']
df = df[df['symbol'].isin(coins)]

# Data validation
print("Data Validation:")
print(f"Total rows: {len(df)}")

# Sort data by timestamp to ensure proper time series analysis
df = df.sort_values(['symbol', 'timestamp'])

# Dictionary to store models and their performance
models = {}
results = []

# Train a separate model for each cryptocurrency
for symbol in df['symbol'].unique():
    print(f"\nTraining model for {symbol}...")

    # Get data for this cryptocurrency
    crypto_data = df[df['symbol'] == symbol].copy()

    # Add technical indicators
    crypto_data = add_technical_indicators(crypto_data)
    if crypto_data.empty:
        print(f"{symbol} has no data after feature engineering. Skipping.")
        continue

    # Make sure there is still data from 2023 after dropna
    if not (crypto_data['timestamp'].dt.year == 2023).any():
        print(f"{symbol} has no 2023 data after dropna. Skipping.")
        continue

    print(f"{symbol} data after dropna: {len(crypto_data)} rows")
    print("Year counts:\n", crypto_data['timestamp'].dt.year.value_counts())

    # Select features (all from t-1) and target (close price at t)
    X = crypto_data[['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support',
                     'close_lag_1', 'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d', 'eth_log_return_lag1']]
    y = crypto_data['close']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- TIME-BASED SPLIT: Train on 2023, Test on 2024+ ---
    crypto_data['year'] = crypto_data['timestamp'].dt.year
    train_mask = crypto_data['year'] == 2023
    test_mask = crypto_data['year'] >= 2024
    X_train = X_scaled[train_mask.values]
    X_test = X_scaled[test_mask.values]
    y_train = y[train_mask.values]
    y_test = y[test_mask.values]

    # --- Naive Baseline R2 (Close t-1 only) ---
    from sklearn.linear_model import LinearRegression
    # Find the index of 'close_lag_1' in X columns
    close_lag_1_idx = X.columns.get_loc('close_lag_1')
    naive_model = LinearRegression()
    if X_train[:, [close_lag_1_idx]].shape[0] == 0:
        print(f"Skipping naive baseline for {symbol} due to empty input.")
        continue
    naive_model.fit(X_train[:, [close_lag_1_idx]], y_train)
    naive_r2 = naive_model.score(X_test[:, [close_lag_1_idx]], y_test)
    print(f"Naive Baseline R2 (Close t-1 only): {naive_r2:.4f}")

    # If not enough data in either set, skip this symbol
    if len(X_train) < 30 or len(X_test) < 10:
        print(f"Not enough data for {symbol} to perform time-based split. Skipping.")
        continue

    # Find optimal alpha for Ridge regression
    ridge_cv = RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_squared_error'
    )
    ridge_cv.fit(X_train, y_train)
    best_alpha_ridge = ridge_cv.alpha_

    # Find optimal alpha for Lasso regression
    lasso_cv = LassoCV(
        alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=TimeSeriesSplit(n_splits=3),
        max_iter=10000
    )
    lasso_cv.fit(X_train, y_train)
    best_alpha_lasso = lasso_cv.alpha_

    # Train Ridge model
    ridge_model = Ridge(alpha=best_alpha_ridge)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_r2 = ridge_model.score(X_test, y_test)

    # Train Lasso model
    lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    lasso_rmse = np.sqrt(lasso_mse)
    lasso_mae = mean_absolute_error(y_test, lasso_pred)
    lasso_r2 = lasso_model.score(X_test, y_test)

    # Choose the better performing model
    if ridge_r2 > lasso_r2:
        model = ridge_model
        model_type = "Ridge"
        mse = ridge_mse
        rmse = ridge_rmse
        mae = ridge_mae
        r2 = ridge_r2
        y_pred = ridge_pred
    else:
        model = lasso_model
        model_type = "Lasso"
        mse = lasso_mse
        rmse = lasso_rmse
        mae = lasso_mae
        r2 = lasso_r2
        y_pred = lasso_pred

    print(f"Naive vs Model R² Improvement: {r2 - naive_r2:.4f}")

    # Store results
    results.append({
        'symbol': symbol,
        'model_type': model_type,
        'alpha': best_alpha_ridge if model_type == "Ridge" else best_alpha_lasso,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mae_pct': (mae / crypto_data['close'].mean()) * 100,  # MAE as percentage of mean price
        'r2': r2,
        'feature_importance': dict(zip(
            ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support',
             'close_lag_1', 'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d', 'eth_log_return_lag1'],
            np.abs(model.coef_)))
    })

    # Store model and scaler
    models[symbol] = {
        'model': model,
        'scaler': scaler,
        'feature_columns': ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support',
                            'close_lag_1', 'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d', 'eth_log_return_lag1'],
        'model_type': model_type,
        'alpha': best_alpha_ridge if model_type == "Ridge" else best_alpha_lasso
    }

    print(f"{symbol} Performance ({model_type}):")
    print(f"Alpha: {best_alpha_ridge if model_type == 'Ridge' else best_alpha_lasso:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.8f}")
    print(f"MAE % of mean price: {(mae / crypto_data['close'].mean()) * 100:.2f}%")
    print(f"R2 Score: {r2:.4f}")

    # Print feature importance
    print("\nTop 10 most important features:")
    feature_importance = pd.DataFrame({
        'feature': ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support',
                    'close_lag_1', 'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d', 'eth_log_return_lag1'],
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))

    # Print some actual predictions vs real values
    print("\nSample predictions vs actual values:")
    sample_size = min(5, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    for idx in sample_indices:
        pred = y_pred[idx]
        actual = y_test.iloc[idx]
        print(f"Predicted: ${pred:.2f}, Actual: ${actual:.2f}, Difference: ${abs(pred-actual):.2f}")

    # --- Volatility Normalization (14-day rolling std of price changes) ---
    test_indices = np.where(test_mask.values)[0]
    test_dates = crypto_data[test_mask]['timestamp'].values
    # Compute rolling std of price changes (close-to-close, 14-day window, up to t-1)
    crypto_data['close_change'] = crypto_data['close'].diff()
    crypto_data['rolling_volatility'] = crypto_data['close_change'].rolling(window=14, min_periods=1).std().shift(1)
    rolling_vol_test = crypto_data.iloc[test_indices]['rolling_volatility'].values

    # --- Dynamic Signal Thresholds ---
    # Try multiple thresholds (percent of price): 1%, 1.5%, 2%, 2.5%, 3%
    thresholds_pct = [0.01, 0.015, 0.02, 0.025, 0.03]
    threshold_results = []
    for pct in thresholds_pct:
        threshold_values = pct * y_test.values  # threshold in price units
        # Volatility filter: signal only if abs(predicted change) > 1.5 * recent volatility
        predicted_change = y_pred - y_test.values
        volatility_filter = np.abs(predicted_change) > (1.5 * rolling_vol_test)
        # Generate signals with this threshold
        signals, signal_strengths = generate_trading_signals(y_test.values, y_pred, 0)  # We'll filter by threshold below
        # Only keep signals where abs(predicted_change) > threshold & passes volatility filter
        filtered_signals = []
        filtered_indices = []
        confidence_scores = []
        for i, (chg, sig, th, vol_pass, strength) in enumerate(zip(predicted_change, signals, threshold_values, volatility_filter, signal_strengths)):
            if (np.abs(chg) > th) and vol_pass:
                filtered_signals.append(sig)
                filtered_indices.append(i)
                confidence_scores.append(abs(strength))
            else:
                filtered_signals.append('HOLD')
                confidence_scores.append(abs(strength))
        # Backtest
        profits = backtest_signals(y_test.values, filtered_signals)
        cumulative_return = np.sum(profits)
        avg_daily_return = np.mean(profits)
        win_trades = [p for p in profits if p > 0]
        loss_trades = [p for p in profits if p < 0]
        num_trades = sum([s != 'HOLD' for s in filtered_signals])
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
        # Directional accuracy: sign(predicted change) vs sign(actual change)
        # For each day except last (since we can't trade on last)
        pred_sign = np.sign(predicted_change[:-1])
        actual_sign = np.sign(np.diff(y_test.values))
        dir_acc = np.mean(pred_sign == actual_sign)
        # Top 25% confidence trades (by abs(predicted change pct)
        abs_strengths = np.abs(signal_strengths)
        top_conf_cutoff = np.percentile(abs_strengths, 75)
        top_conf_indices = [i for i, s in enumerate(abs_strengths) if s >= top_conf_cutoff]
        top_conf_signals = [filtered_signals[i] if i in top_conf_indices else 'HOLD' for i in range(len(filtered_signals))]
        profits_top_conf = backtest_signals(y_test.values, top_conf_signals)
        num_top_conf_trades = sum([s != 'HOLD' for s in top_conf_signals])
        win_rate_top_conf = len([p for p in profits_top_conf if p > 0]) / num_top_conf_trades if num_top_conf_trades > 0 else 0
        sharpe_top_conf = np.mean(profits_top_conf) / np.std(profits_top_conf) if np.std(profits_top_conf) > 0 else 0
        threshold_results.append({
            'threshold_pct': pct,
            'cumulative_return': cumulative_return,
            'avg_daily_return': avg_daily_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'directional_accuracy': dir_acc,
            'num_trades': num_trades,
            'cumulative_return_top25conf': np.sum(profits_top_conf),
            'win_rate_top25conf': win_rate_top_conf,
            'sharpe_top25conf': sharpe_top_conf,
            'num_trades_top25conf': num_top_conf_trades
        })
        # For the lowest threshold, save the signals for visualization/saving
        if pct == thresholds_pct[0]:
            save_signal_df = pd.DataFrame({
                'timestamp': test_dates,
                'actual': y_test.values,
                'predicted': y_pred,
                'signal': filtered_signals,
                'confidence_score': confidence_scores,
                'profit': profits
            })
            save_signal_df['cumulative_profit'] = np.cumsum(profits)

    # Print threshold comparison
    print(f"\nThreshold comparison for {symbol}:")
    for res in threshold_results:
        print(f"Threshold: {res['threshold_pct']*100:.2f}% | Trades: {res['num_trades']} | Win Rate: {res['win_rate']*100:.1f}% | Sharpe: {res['sharpe_ratio']:.2f} | Avg Return: {res['avg_daily_return']:.4f} | Dir. Acc: {res['directional_accuracy']*100:.1f}% | Top25% Trades: {res['num_trades_top25conf']} | Top25% WinRate: {res['win_rate_top25conf']*100:.1f}% | Top25% Sharpe: {res['sharpe_top25conf']:.2f}")

    # Save backtest results for the lowest threshold (1%) with confidence
    save_signal_df.to_csv(os.path.join(project_root, "graphs", f"{symbol}_signals.csv"), index=False)
    print(f"Backtest results added to {symbol}_signals.csv")

    # 5. Cumulative Profit Plot (for first threshold)
    plt.figure(figsize=(10, 5))
    plt.plot(save_signal_df['timestamp'], save_signal_df['cumulative_profit'], label='Cumulative Profit', color='green')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'{symbol} - Backtested Cumulative Profit (Threshold={thresholds_pct[0]*100:.1f}%)')
    plt.xlabel('Date')
    plt.ylabel('Profit ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "graphs", f"{symbol}_cumulative_profit.png"))
    plt.close()

    # --- Visualization ---
    # 1. Predicted vs Actual
    plt.figure(figsize=(8, 5))
    plt.plot(y_test.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.title(f'{symbol} - Predicted vs Actual Close Price')
    plt.xlabel('Sample')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "graphs", f"{symbol}_pred_vs_actual.png"))
    plt.close()

    # 2. Residuals Plot
    residuals = y_test.values - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(residuals)), residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{symbol} - Residuals (Actual - Predicted)')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "graphs", f"{symbol}_residuals.png"))
    plt.close()

    # 3. Feature Importance Bar Chart
    feature_importance = pd.DataFrame({
        'feature': ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support',
                    'close_lag_1', 'close_lag_2', 'volume_lag_1', 'rolling_volatility_7d', 'eth_log_return_lag1'],
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15), palette='viridis')
    plt.title(f'{symbol} - Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "graphs", f"{symbol}_feature_importance.png"))
    plt.close()

    # 4. Rug Pull Risk (if present)
    if 'rug_pull_risk' in X.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(crypto_data.index, crypto_data['rug_pull_risk'], label='Rug Pull Risk', color='orange')
        plt.title(f'{symbol} - Rug Pull Risk Over Time')
        plt.xlabel('Index')
        plt.ylabel('Rug Pull Risk')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, "graphs", f"{symbol}_rug_pull_risk.png"))
        plt.close()

# Create summary DataFrame

results_df = pd.DataFrame([{
    'symbol': r['symbol'],
    'model_type': r['model_type'],
    'alpha': r['alpha'],
    'mse': r['mse'],
    'rmse': r['rmse'],
    'mae': r['mae'],
    'mae_pct': r['mae_pct'],
    'r2': r['r2'],
    # Additional columns will be filled below if available
} for r in results])

# Save summary DataFrame to CSV in the data directory
results_df.to_csv(os.path.join(project_root, "data", "model_performance_summary.csv"), index=False)

# Check if results_df is empty before plotting or further processing
if results_df.empty:
    print("No models were trained successfully. Exiting.")
    sys.exit()

# Optionally, aggregate threshold_results for summary (pick, e.g., 1% threshold)
threshold_summary = []
for symbol in df['symbol'].unique():
    try:
        sfile = os.path.join(project_root, "graphs", f"{symbol}_signals.csv")
        sdata = pd.read_csv(sfile)
        # You could load threshold_results from a file if you saved, or summarize here
        # For now, leave as is.
    except Exception:
        pass

print("\nSummary of all models:")
print(results_df.to_string(index=False))

# Save all models
joblib.dump(models, os.path.join(project_root, "models", "crypto_models.joblib"))
print("\nAll models saved as 'crypto_models.joblib'")

