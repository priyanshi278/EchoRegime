# Autonomous Adaptive Portfolio & Risk Management Engine

## 💡 The Idea
Traditional portfolio management systems are static and reactive. They often fail to protect capital during rapid market downturns and struggle to explain their decisions to users.

**Our Solution**: An intelligent, autonomous engine that dynamically adjusts asset allocation based on real-time market regime detection. It doesn't just manage risk; it *explains* its decisions using Explainable AI (XAI) and allows users to stress-test the system via an interactive "Crisis Lab".

## 🚀 Key Features

### 1. **Dynamic Regime Detection**
   - Automatically classifies the market into four regimes: **Bullish, Bearish, High Volatility, Crash**.
   - Uses a combination of Trend (SMA 200) and Volatility (VIX) indicators.

### 2. **Adaptive Asset Allocation**
   - **Bullish**: High Equity exposure (Aggressive).
   - **High Volatility/Bearish**: Shifts to Bonds and Cash (Defensive).
   - **Crash**: Maximum protection (Cash heavy).

### 3. **Risk Management Engine**
   - **Volatility Targeting**: Scales position sizes inversely to market volatility.
   - **Drawdown Control**: Hard stop-loss mechanisms to preserve capital.

### 4. **🧠 Transparent Brain (Explainable AI)**
   - **Powered by SHAP**: Uses Shapley Additive Explanations to visualize *why* the AI chose a specific regime.
   - **Generative Explanations**: Integrates with **Groq (Llama-3)** to provide natural language summaries of market conditions.

### 5. **🧪 Crisis Lab (Interactive Stress Testing)**
   - Allows users to inject hypothetical market shocks (e.g., -10% drop, 2x volatility spike).
   - Demonstrates in real-time how the engine would rebalance to protect the portfolio.

## 🏗️ Architecture

The project follows a modular microservices-ready architecture:

```
├── app/
│   ├── core/               # Core Logic Modules
│   │   ├── data_loader.py      # Fetches market data (yfinance)
│   │   ├── regime_detector.py  # Classifies market state
│   │   ├── allocation_engine.py# Determines weights
│   │   ├── risk_manager.py     # Applies risk controls
│   │   ├── backtester.py       # Simulation engine
│   │   ├── explainer.py        # GenAI (Groq) integration
│   │   └── xai_engine.py       # SHAP interpretation
│   ├── api/                # FastAPI Backend
│   │   └── main.py             # REST Endpoints
│   └── ui/                 # Streamlit Frontend
│       └── dashboard.py        # Interactive Dashboard
├── requirements.txt    # Dependencies
├── run_app.bat         # One-click launcher
└── .env                # API Keys
```

## 🛠️ Tech Stack

-   **Language**: Python 3.9+
-   **Frontend**: Streamlit (Interactive Dashboard)
-   **Backend**: FastAPI (Rest API)
-   **Data Source**: yfinance (Yahoo Finance API)
-   **Machine Learning / AI**:
    -   `scikit-learn`: Random Forest (Surrogate Model)
    -   `shap`: Explainable AI
    -   `groq`: Generative AI (Llama-3) for Natural Language Explanations
-   **Data Science**: `pandas`, `numpy`
-   **Visualization**: `plotly` (Interactive Charts), `matplotlib` (SHAP plots)

## ⚡ How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Key**:
    -   Open `.env` file.
    -   Add your Groq API Key: `GROQ_API_KEY=your_key_here`

3.  **Launch Dashboard**:
    ```bash
    streamlit run app/ui/dashboard.py
    ```
