import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical data for given tickers.
    """
    # multi-level columns=True by default in recent yfinance if multiple tickers or explicit list
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Handle MultiIndex or standard columns
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            return data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            return data['Close']
    
    # Flat columns
    if 'Adj Close' in data.columns:
        return data['Adj Close']
    elif 'Close' in data.columns:
        if data.empty:
            raise ValueError(f"No data found for ticker(s) {tickers}. Check symbol or date range.")

    return data['Close']
        
    return data

def calculate_features(data):
    """
    Calculates technical indicators ensuring no data leakage (using shift).
    """
    features = pd.DataFrame(index=data.index)
    
    # Calculate returns
    features['Returns'] = data.pct_change()
    
    # Calculate Log Returns
    features['Log_Returns'] = np.log(data / data.shift(1))
    
    # Rolling Volatility (21 days ~ 1 month)
    features['Volatility'] = features['Log_Returns'].rolling(window=21).std() * np.sqrt(252)
    
    # Simple Moving Averages
    features['SMA_50'] = data.rolling(window=50).mean()
    features['SMA_200'] = data.rolling(window=200).mean()
    
    # Momentum (Returns over past 3 months)
    features['Momentum_3M'] = data.pct_change(periods=63)
    
    # Drawdown
    rolling_max = data.rolling(window=252, min_periods=1).max()
    features['Drawdown'] = (data - rolling_max) / rolling_max
    
    # Market State features
    # 1 if Price > SMA200, else 0 (Long term Trend)
    features['Trend'] = np.where(data > features['SMA_200'], 1, 0)
    
    return features.dropna()
