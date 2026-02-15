import pandas as pd
import numpy as np

class RegimeDetector:
    def __init__(self, high_vol_threshold=0.20, crash_threshold=0.40):
        self.high_vol_threshold = high_vol_threshold
        self.crash_threshold = crash_threshold

    def detect_regime(self, market_data):
        """
        Detects market regime based on volatility and trend.
        Returns:
            - 'Bullish'
            - 'Bearish'
            - 'High Volatility'
            - 'Crash'
        """
        # Ensure we have the necessary columns
        if 'Volatility' not in market_data.columns or 'SMA_200' not in market_data.columns:
            raise ValueError("Market data must contain 'Volatility' and 'SMA_200'")
        
        last_row = market_data.iloc[-1]
        price = last_row.name if isinstance(last_row.name, (int, float)) else last_row.iloc[0]
 # handling if price is in column 0 or index
        
        # We need the actual price to compare with SMA, but data_loader returns features. 
        # Let's assume market_data passed here has 'Price' or we use the Trend feature directly.
        # Actually data_loader returns 'Trend' (1 or 0). 
        
        volatility = last_row['Volatility']
        trend_up = last_row['Trend'] == 1
        
        if volatility > self.crash_threshold:
            return "Crash"
        elif volatility > self.high_vol_threshold:
            return "High Volatility"
        elif trend_up:
            return "Bullish"
        else:
            return "Bearish"
