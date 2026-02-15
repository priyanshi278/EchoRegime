import pandas as pd
import numpy as np
from app.core.data_loader import fetch_data, calculate_features
from app.core.regime_detector import RegimeDetector
from app.core.allocation_engine import AllocationEngine
from app.core.risk_manager import RiskManager

class Backtester:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        self.results = []
        
    def load_data(self):
        # We need a benchmark/asset to trade. Let's assume 'ticker' is the Equity part (e.g. SPY).
        # We also need Bond data (e.g. TLT) but for simplicity we simulate returns or fetch multiple.
        # For this hackathon scope, let's just fetch the main equity ticker and simulate the "Portfolio" 
        # as a mix of this Ticker and Cash (or a proxy for Bonds).
        
        # Fetch Equity Data
        self.data = fetch_data([self.ticker], self.start_date, self.end_date)
        if hasattr(self.data, 'columns') and isinstance(self.data.columns, pd.MultiIndex):
             self.data = self.data.xs('Adj Close', axis=1, level=0) 
        
        # Ensure it's a Series or DataFrame with the ticker column
        if isinstance(self.data, pd.DataFrame) and self.ticker in self.data.columns:
            self.data = self.data[self.ticker]
            
        self.features = calculate_features(self.data)
        
    def run(self, use_risk_engine=True):
        regime_detector = RegimeDetector()
        allocator = AllocationEngine()
        risk_manager = RiskManager()
        
        # Portfolio Value
        portfolio_value = 10000.0
        portfolio_values = [portfolio_value]
        
        # Logic Loop
        # We iterate through the features. 
        # IMPORTANT: 'features' index aligns with 'data'.
        
        for i in range(len(self.features)):
            # Current State (at close of day i, needed for decision at i+1 open or close)
            # We use data available up to i to make decision for i+1 returns.
            
            if i < 200: # Warm up for SMA
                portfolio_values.append(portfolio_value)
                continue
                
            current_date = self.features.index[i]
            # row = self.features.iloc[i] # Features for today
            
            # Slice data up to today to simulate "live" data for detection
            # Actually we pre-calculated features so we just assume no look-ahead in `calculate_features`.
            
            # 1. Detect Regime
            # We pass a window of data or just current features if they are sufficient
            # Our detect_regime uses 'Volatility' and 'Trend' from the row.
            
            # Construct a mini-dataframe for the detector (it expects a df with columns)
            # current_market_data = self.features.iloc[:i+1] # potentially slow in loop
            # Optimization: just pass the row if logic allows. 
            # Our logic: `last_row = market_data.iloc[-1]`
            
            regime = regime_detector.detect_regime(self.features.iloc[:i+1])
            
            # 2. Allocate
            allocation = allocator.get_allocation(regime)
            
            # 3. Risk Management
            if use_risk_engine:
                current_vol = self.features['Volatility'].iloc[i]
                current_dd = self.features['Drawdown'].iloc[i]
                allocation, mod = risk_manager.apply_risk_controls(current_vol, current_dd, allocation)
            
            # 4. Simulate Return for *Next Day*
            if i + 1 < len(self.features):
                next_ret = self.features['Returns'].iloc[i+1]
                
                # Portfolio Return = EquityWeight * EquityReturn + BondWeight * BondReturn + CashWeight * 0
                # Simulating Bond Return as 0.0 for simplicity or small fixed rate 0.02/252
                bond_ret = 0.02 / 252.0 
                cash_ret = 0.0 
                
                port_ret = (allocation['Equity'] * next_ret) + (allocation['Bonds'] * bond_ret) + (allocation['Cash'] * cash_ret)
                
                portfolio_value *= (1 + port_ret)
                portfolio_values.append(portfolio_value)
                
                # Store state for visualization
                self.results.append({
                    'Date': self.features.index[i+1],
                    'Value': portfolio_value,
                    'Regime': regime,
                    'Equity_Weight': allocation['Equity'],
                    'Bonds_Weight': allocation['Bonds'],
                    'Cash_Weight': allocation['Cash']
                })
                
        return pd.DataFrame(self.results).set_index('Date')

    def calculate_metrics(self, strategy_results):
        if strategy_results.empty:
            return {}
            
        returns = strategy_results['Value'].pct_change().dropna()
        
        total_return = (strategy_results['Value'].iloc[-1] / strategy_results['Value'].iloc[0]) - 1
        cagr = (1 + total_return) ** (252 / len(strategy_results)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.02) / vol if vol > 0 else 0
        
        # Max Drawdown
        cum_max = strategy_results['Value'].cummax()
        drawdown = (strategy_results['Value'] - cum_max) / cum_max
        max_dd = drawdown.min()
        
        return {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        }
