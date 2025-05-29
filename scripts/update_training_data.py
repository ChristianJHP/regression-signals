import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def fetch_latest_data(symbols, start_date='2025-05-25'):
    """Fetch latest data for given symbols from start_date to current date."""
    end_date = datetime.now()
    
    all_data = []
    failed_symbols = []
    
    for symbol in symbols:
        try:
            # Add -USD suffix for Yahoo Finance
            ticker = f"{symbol}-USD"
            print(f"Fetching data for {symbol} from {start_date} to {end_date.strftime('%Y-%m-%d')}")
            
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                print(f"No data found for {symbol}")
                failed_symbols.append(symbol)
                continue
                
            # Reset index to make Date a column
            data = data.reset_index()
            
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
            print(f"Successfully fetched {len(data)} rows for {symbol}")
            
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

def main():
    # List of symbols to fetch
    symbols = ['ETH', 'BNB', 'SOL', 'AVAX', 'MATIC', 'ADA', 'INJ', 'RUNE', 'FET', 'RNDR', 'KAS', 'TWT', 'ARB', 'BPX']
    
    # Fetch latest data from 2023 to current date
    data, failed_symbols = fetch_latest_data(symbols, start_date='2023-01-01')
    if data is None:
        print("Failed to fetch any data. Exiting.")
        return
    
    # Save to CSV
    output_path = os.path.join(project_root, 'data', 'crypto_ohlcv.csv')
    data.to_csv(output_path, index=False)
    print(f"\nSaved {len(data)} rows of data to {output_path}")
    
    if failed_symbols:
        print(f"\nFailed to fetch data for: {failed_symbols}")
    
    # Print data summary
    print("\nData Summary:")
    print(f"Date Range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print("\nRows per symbol:")
    print(data['symbol'].value_counts())

if __name__ == "__main__":
    main() 