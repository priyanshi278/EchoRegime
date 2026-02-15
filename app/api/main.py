from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.backtester import Backtester
from app.core.explainer import Explainer
import pandas as pd

app = FastAPI(title="Autonomous Adaptive Portfolio Engine")

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

@app.post("/backtest")
def run_backtest(request: BacktestRequest):
    try:
        # Run Strategy (With Risk Engine)
        backtester = Backtester(request.ticker, request.start_date, request.end_date)
        backtester.load_data()
        results = backtester.run(use_risk_engine=True)
        metrics = backtester.calculate_metrics(results)
        
        # Run Benchmark (Buy & Hold / Without Risk Engine)
        results_bench = backtester.run(use_risk_engine=False)
        metrics_bench = backtester.calculate_metrics(results_bench)
        
        # Generate Explanation
        explainer = Explainer()
        # Get last state for explanation
        if not results.empty:
            last_row = results.iloc[-1]
            last_regime = last_row['Regime']
            last_alloc = {
                'Equity': last_row['Equity_Weight'],
                'Bonds': last_row['Bonds_Weight'],
                'Cash': last_row['Cash_Weight']
            }
            explanation = explainer.explain(last_regime, last_alloc, metrics)
        else:
            explanation = "No data available."

        # Convert simple types for JSON
        def clean_nan(obj):
            if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
                return None
            return obj

        return {
            "metrics_strategy": {k: clean_nan(v) for k, v in metrics.items()},
            "metrics_benchmark": {k: clean_nan(v) for k, v in metrics_bench.items()},
            "data_strategy": results.reset_index().to_dict(orient='records'),
            "data_benchmark": results_bench.reset_index().to_dict(orient='records'),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "System Operational"}
