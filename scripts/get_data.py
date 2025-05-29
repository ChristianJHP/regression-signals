import requests
import pandas as pd
from datetime import datetime, timedelta
import time

API_KEY = "A5gXYy_wbQjAnVDmIfHpoy9VlD1q9AAx"

def get_crypto_data(symbol, from_date="2023-01-01", to_date="2025-05-25", timespan="day"):
    """
    Fetch cryptocurrency data with better error handling and validation
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/range/1/{timespan}/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }
    
    max_retries = 3
    base_delay = 5  # Base delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 429:  # Too Many Requests
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limit hit. Waiting {delay} seconds before retry...")
                time.sleep(delay)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if "results" not in data:
                print(f"Error: No results found for {symbol}")
                print(f"API Response: {data}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data["results"])
            
            # Validate data
            if len(df) == 0:
                print(f"Error: Empty dataset for {symbol}")
                return pd.DataFrame()
                
            # Convert timestamp
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            
            # Rename columns
            df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
                'v': 'volume', 't': 'timestamp'
            }, inplace=True)
            
            # Add symbol
            df['symbol'] = symbol
            
            # Calculate daily volatility
            df['daily_return'] = df['close'].pct_change()
            df['daily_volatility'] = df['daily_return'].abs()
            
            # Detect extreme price movements
            df['extreme_move'] = df['daily_volatility'] > 0.2  # 20% daily move
            
            # Calculate volume spikes
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_spike'] = df['volume'] > (df['volume_ma'] * 3)  # 3x average volume
            
            # Print data summary with volatility metrics
            print(f"\n{symbol} Data Summary:")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Number of records: {len(df)}")
            print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            print(f"Average daily volatility: {df['daily_volatility'].mean():.2%}")
            print(f"Max daily volatility: {df['daily_volatility'].max():.2%}")
            print(f"Number of extreme moves (>20%): {df['extreme_move'].sum()}")
            print(f"Number of volume spikes (>3x avg): {df['volume_spike'].sum()}")
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 
                      'daily_return', 'daily_volatility', 'extreme_move', 'volume_spike']]
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Error fetching data for {symbol}: {str(e)}")
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                return pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error for {symbol}: {str(e)}")
            return pd.DataFrame()

# List of cryptocurrencies to fetch - including some high-risk ones
cryptos = [
    'ETH',  # Ethereum - Gold standard altcoin, L1 benchmark
    'BNB',  # BNB Chain - Exchange coin, whale-driven
    'SOL',  # Solana - Volatile AF, great for momentum models
    'AVAX', # Avalanche - Consistent volume, fast pump cycles
    'MATIC',# Polygon - Your data source, widely integrated
    'ADA',  # Cardano - Good for trend following (slow mover)
    'INJ',  # Injective - Pumps hard, TA-friendly, whale games
    'RUNE', # THORChain - Cyclical, great for MACD-based setups
    'FET',  # Fetch.AI - AI hype-sensitive, solid RSI signals
    'RNDR', # Render - Retail + AI narrative = easy swing plays
    'KAS',  # Kaspa - Low cap, high signal/noise ratio
    'TWT',  # Trust Wallet - Pumps off exchange/news events
    'ARB',  # Arbitrum - Speculative L2, very bot-active
    'BPX',  # Bitcoin Private - Bitcoin-like, high-risk, high-reward
]

# Get data for all cryptocurrencies
all_data = []
for crypto in cryptos:
    print(f"\nFetching data for {crypto}...")
    df = get_crypto_data(crypto)
    if not df.empty:
        all_data.append(df)
    time.sleep(10)  # Increased delay between requests to avoid rate limits

# Combine all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Final validation
    print("\nFinal Data Validation:")
    print(f"Total records: {len(combined_df)}")
    print("\nRecords per cryptocurrency:")
    print(combined_df['symbol'].value_counts())
    
    # Analyze volatility patterns
    print("\nVolatility Analysis:")
    volatility_summary = combined_df.groupby('symbol').agg({
        'daily_volatility': ['mean', 'max', 'std'],
        'extreme_move': 'sum',
        'volume_spike': 'sum'
    }).round(4)
    print(volatility_summary)
    
    # Save to CSV
    combined_df.to_csv("crypto_ohlcv.csv", index=False)
    print(f"\nSaved data to crypto_ohlcv.csv")
else:
    print("No data was fetched successfully")
