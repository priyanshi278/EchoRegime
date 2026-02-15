class RiskManager:
    def __init__(self, target_vol=0.15, max_drawdown=0.20):
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown

    def apply_risk_controls(self, current_vol, current_drawdown, proposed_allocation):
        """
        Applies volatility targeting and drawdown protection.
        Returns modified allocation.
        """
        modifier = 1.0
        
        # Volatility Targeting
        if current_vol > self.target_vol:
            # Reduce exposure ratio
            vol_scalar = self.target_vol / current_vol
            modifier *= vol_scalar
        
        # Drawdown Protection (Stop Loss)
        if current_drawdown > self.max_drawdown:
            # Cut exposure significantly (e.g., to 0 or 10%)
            modifier = 0.0 # complete exit
            
        # Apply modifier to Equity and Bonds (features that have risk)
        # Cash/Gold might be considered safe, but usually we just scale down risk assets and move to cash.
        
        new_allocation = proposed_allocation.copy()
        for asset in ['Equity', 'Bonds']:
            if asset in new_allocation:
                new_allocation[asset] *= modifier
        
        # Re-balance the remainder to Cash
        total_risk_weight = sum([new_allocation.get(a, 0) for a in ['Equity', 'Bonds']])
        new_allocation['Cash'] = 1.0 - total_risk_weight
        
        return new_allocation, modifier
