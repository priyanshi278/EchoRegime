class AllocationEngine:
    def __init__(self):
        pass
    
    def get_allocation(self, regime):
        """
        Returns asset allocation weights based on the market regime.
        The assets are roughly: [Equity, Bonds, Gold/Cash]
        """
        if regime == "Bullish":
            # Risk On: High Equity, Moderate Bonds, Low Cash
            return {"Equity": 0.70, "Bonds": 0.20, "Cash": 0.10}
        
        elif regime == "Bearish":
            # Risk Off: Low Equity, High Bonds, Moderate Cash
            return {"Equity": 0.30, "Bonds": 0.50, "Cash": 0.20}
        
        elif regime == "High Volatility":
            # Defensive: Very Low Equity, High Cash
            return {"Equity": 0.10, "Bonds": 0.40, "Cash": 0.50}
        
        elif regime == "Crash":
            # Capital Preservation: Max Cash
            return {"Equity": 0.00, "Bonds": 0.20, "Cash": 0.80}
        
        else:
            # Default Balanced
            return {"Equity": 0.50, "Bonds": 0.40, "Cash": 0.10}
