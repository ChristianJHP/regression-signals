import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import matplotlib.pyplot as plt
import os

# Parsing and cleaning data
def parse_and_clean_data(csv_data):
    df = pd.read_csv(csv_data, sep=';')
    df.columns = df.columns.str.strip().str.replace('"', '')
    df = df.apply(lambda x: x.str.strip().str.replace('"', '') if x.dtype == "object" else x)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    # Interpolate to daily data (temporary placeholder)
    date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')
    df = df.set_index('timestamp').reindex(date_range).interpolate(method='linear').reset_index()
    df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

# Creating features
def create_features(df):
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = SMAIndicator(df['close'], window=5).sma_indicator()
    df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    return df

# Training the model
def train_model(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 'bb_high', 'bb_low', 'macd', 'macd_signal', 'atr']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Training Target Distribution: {np.bincount(y_train)}")
    print(f"Test Target Distribution: {np.bincount(y_test)}")
    
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Precision: {precision:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print(f"Prediction Distribution (Test Set): {np.bincount(y_pred)}")
    
    return model, X_test, y_test, y_pred

# Simulating trading strategy
def simulate_trading(df, predictions, X_test):
    test_df = df.loc[X_test.index].copy()
    test_df['predictions'] = predictions
    
    initial_capital = 10000
    position = 0
    capital = initial_capital
    trades = []
    
    for i in range(len(test_df)-1):
        if test_df['predictions'].iloc[i] == 1 and position == 0:
            shares = capital / test_df['close'].iloc[i]
            position = 1
            trades.append(('buy', test_df['timestamp'].iloc[i], test_df['close'].iloc[i], shares))
        elif test_df['predictions'].iloc[i] == 0 and position == 1:
            capital = shares * test_df['close'].iloc[i]
            position = 0
            trades.append(('sell', test_df['timestamp'].iloc[i], test_df['close'].iloc[i], capital))
    
    if position == 1:
        capital = shares * test_df['close'].iloc[-1]
        trades.append(('sell', test_df['timestamp'].iloc[-1], test_df['close'].iloc[-1], capital))
    
    print(f"\nFinal Capital: ${capital:.2f}")
    print(f"Return: {((capital - initial_capital) / initial_capital * 100):.2f}%")
    return trades, capital

# Plotting results
def plot_results(df, X_test, y_pred):
    test_df = df.loc[X_test.index].copy()
    test_df['predictions'] = y_pred
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['timestamp'], test_df['close'], label='Actual Close Price', color='blue')
    plt.plot(test_df[test_df['predictions'] == 1]['timestamp'], 
             test_df[test_df['predictions'] == 1]['close'], 
             '^', color='green', label='Buy Signal', markersize=10)
    plt.plot(test_df[test_df['predictions'] == 0]['timestamp'], 
             test_df[test_df['predictions'] == 0]['close'], 
             'v', color='red', label='Sell Signal', markersize=10)
    
    plt.title('Dogecoin Daily Price with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('doge_daily_trading_signals.png')
    plt.close()

# Main execution
def main():
    file_path = os.path.expanduser("~/Desktop/crypto/scripts/longer_frame/DOGE_All_graph_coinmarketcap.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}")
    
    df = parse_and_clean_data(file_path)
    df = create_features(df)
    model, X_test, y_test, y_pred = train_model(df)
    trades, final_capital = simulate_trading(df, y_pred, X_test)
    
    print("\nTrades Executed:")
    for trade in trades:
        print(f"{trade[0].capitalize()} at {trade[1]}: Price ${trade[2]:.6f}, {'Shares' if trade[0] == 'buy' else 'Capital'}: {trade[3]:.2f}")
    
    plot_results(df, X_test, y_pred)
    print("\nPlot saved as 'doge_daily_trading_signals.png'")

if __name__ == "__main__":
    main()