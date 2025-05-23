import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def add_technical_indicators(df):
    """
    Add technical indicators using only past data (t-1) to predict current price (t).
    This function ensures NO data leakage: all features are calculated using only information available up to t-1.
    """
    # Sort by timestamp to ensure proper time order
    df = df.sort_values('timestamp')
    
    # Calculate log returns using previous day's close
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate 20-day EMA using only past data
    # We'll calculate it manually to ensure we only use past data
    def calculate_ema(data, span):
        alpha = 2 / (span + 1)
        ema = [data.iloc[0]]  # Initialize with first value by position
        for i in range(1, len(data)):
            ema.append(alpha * data.iloc[i] + (1 - alpha) * ema[i-1])
        return pd.Series(ema, index=data.index)
    
    # Calculate EMA using only past data
    df['ema_20'] = df.groupby('symbol')['close'].transform(
        lambda x: calculate_ema(x, 20).shift(1)
    )
    
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
    
    # Drop the first rows where we don't have t-1 data for all features
    df = df.dropna()
    
    # Validate that features do not correlate with current day's close
    print("\nValidating feature timing (correlation with current day's close):")
    print("=" * 80)
    for feature in ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support']:
        corr = df[feature].corr(df['close'])
        if abs(corr) > 0.5:
            print(f"WARNING: {feature} shows high correlation ({corr:.3f}) with current day's close price (possible leakage)")
        else:
            print(f"OK: {feature} shows low correlation ({corr:.3f}) with current day's close price")
    print("=" * 80)
    
    return df

# --- Trading Signal Generation ---
def generate_trading_signals(y_test, y_pred, threshold):
    """
    Generate trading signals ('BUY', 'SELL', 'HOLD') based on predicted vs actual prices.
    A 'BUY' is signaled if predicted > actual + threshold,
    a 'SELL' if predicted < actual - threshold,
    otherwise 'HOLD'.
    """
    signals = []
    for actual, predicted in zip(y_test, y_pred):
        diff = predicted - actual
        if diff > threshold:
            signals.append('BUY')
        elif diff < -threshold:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

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
df = pd.read_csv("crypto_ohlcv.csv")
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
    
    # Select features (all from t-1) and target (close price at t)
    X = crypto_data[['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support']]
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
        'feature_importance': dict(zip(['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support'], np.abs(model.coef_)))
    })
    
    # Store model and scaler
    models[symbol] = {
        'model': model,
        'scaler': scaler,
        'feature_columns': ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support'],
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
        'feature': ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support'],
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

    # --- Trading Signal Logic ---
    signal_threshold = 2 * mae
    trading_signals = generate_trading_signals(y_test, y_pred, signal_threshold)
    signal_df = pd.DataFrame({
        'timestamp': crypto_data[test_mask]['timestamp'].values,
        'actual': y_test.values,
        'predicted': y_pred,
        'signal': trading_signals
    })

    # --- Backtest ---
    profits = backtest_signals(y_test.values, trading_signals)
    cumulative_return = np.sum(profits)
    avg_daily_return = np.mean(profits)
    print(f"Cumulative return for {symbol}: ${cumulative_return:.2f}")
    print(f"Average daily return: ${avg_daily_return:.4f}")

    # Save backtest results
    signal_df['profit'] = profits
    signal_df['cumulative_profit'] = np.cumsum(profits)
    signal_df.to_csv(f"graphs/{symbol}_signals.csv", index=False)
    print(f"Backtest results added to {symbol}_signals.csv")

    # 5. Cumulative Profit Plot
    plt.figure(figsize=(10, 5))
    plt.plot(signal_df['timestamp'], signal_df['cumulative_profit'], label='Cumulative Profit', color='green')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'{symbol} - Backtested Cumulative Profit')
    plt.xlabel('Date')
    plt.ylabel('Profit ($)')
    plt.tight_layout()
    plt.savefig(f'graphs/{symbol}_cumulative_profit.png')
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
    plt.savefig(f'graphs/{symbol}_pred_vs_actual.png')
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
    plt.savefig(f'graphs/{symbol}_residuals.png')
    plt.close()

    # 3. Feature Importance Bar Chart
    feature_importance = pd.DataFrame({
        'feature': ['log_return', 'ema_20', 'macd_hist', 'daily_range_pct', 'volume_change', 'price_to_support'],
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15), palette='viridis')
    plt.title(f'{symbol} - Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'graphs/{symbol}_feature_importance.png')
    plt.close()

    # 4. Rug Pull Risk (if present)
    if 'rug_pull_risk' in X.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(crypto_data.index, crypto_data['rug_pull_risk'], label='Rug Pull Risk', color='orange')
        plt.title(f'{symbol} - Rug Pull Risk Over Time')
        plt.xlabel('Index')
        plt.ylabel('Rug Pull Risk')
        plt.tight_layout()
        plt.savefig(f'graphs/{symbol}_rug_pull_risk.png')
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
    'r2': r['r2']
} for r in results])


print("\nSummary of all models:")
print(results_df.to_string(index=False))

# Save all models
joblib.dump(models, 'crypto_models.joblib')
print("\nAll models saved as 'crypto_models.joblib'")

# --- Summary Visualization ---
plt.figure(figsize=(12, 6))
results_df_sorted = results_df.sort_values('r2', ascending=False)
sns.barplot(x='r2', y='symbol', data=results_df_sorted, palette='coolwarm')
plt.title('R² Score by Cryptocurrency Model')
plt.xlabel('R² Score')
plt.ylabel('Cryptocurrency')
plt.tight_layout()
plt.savefig('graphs/model_r2_scores.png')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='mae_pct', y='symbol', data=results_df_sorted, palette='magma')
plt.title('Mean Absolute Error (% of Price) by Cryptocurrency Model')
plt.xlabel('MAE (%)')
plt.ylabel('Cryptocurrency')
plt.tight_layout()
plt.savefig('graphs/model_mae_percent.png')
plt.show()
