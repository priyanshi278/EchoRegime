from fastapi import FastAPI
from app.routes import regime, allocation, risk, backtest

app = FastAPI()

app.include_router(regime.router)
app.include_router(allocation.router)
app.include_router(risk.router)
app.include_router(backtest.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}