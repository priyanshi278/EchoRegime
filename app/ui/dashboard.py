import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

st.set_page_config(page_title="Autonomous Portfolio Engine", layout="wide")

st.title("🧠 Autonomous Adaptive Portfolio & Risk Engine")
st.markdown("### AI-Driven | Regime-Aware | Volatility-Targeting")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# Session State Initialization
if 'sim_run' not in st.session_state:
    st.session_state.sim_run = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'backtester_features' not in st.session_state:
    st.session_state.backtester_features = None

if st.sidebar.button("Run Simulation"):
    st.session_state.sim_run = True
    # Reset crisis state on new run
    st.session_state.crisis_simulated = False
    st.session_state.crisis_msg = ""
    
    with st.spinner("Crunching numbers... Detecting Regimes... Allocating Capital..."):
        try:
            # FORCE LOCAL EXECUTION for Hackathon Demo
            # This ensures XAI (SHAP) has access to the raw DataFrames/Models which is hard via API.
            from app.core.backtester import Backtester
            from app.core.explainer import Explainer
            
            backtester = Backtester(ticker, str(start_date), str(end_date))
            backtester.load_data()
            
            # Store features in session state for XAI
            st.session_state.backtester_features = backtester.features
            
            res_strat = backtester.run(use_risk_engine=True)
            met_strat = backtester.calculate_metrics(res_strat)
            
            res_bench = backtester.run(use_risk_engine=False)
            met_bench = backtester.calculate_metrics(res_bench)
            
            explainer = Explainer()
            if not res_strat.empty:
                last_row = res_strat.iloc[-1]
                explanation = explainer.explain(last_row['Regime'], 
                                                {'Equity': last_row['Equity_Weight'], 'Bonds': last_row['Bonds_Weight'], 'Cash': last_row['Cash_Weight']},
                                                met_strat)
            else:
                explanation = "No data."
            
            st.session_state.data = {
                "metrics_strategy": met_strat,
                "metrics_benchmark": met_bench,
                "data_strategy": res_strat.reset_index().to_dict(orient='records'),
                "data_benchmark": res_bench.reset_index().to_dict(orient='records'),
                "explanation": explanation
            }
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.sim_run = False

# Main Rendering Logic
if st.session_state.sim_run and st.session_state.data:
    data = st.session_state.data
    
    # Process Data
    df_strat = pd.DataFrame(data['data_strategy'])
    df_bench = pd.DataFrame(data['data_benchmark'])
    
    # Handle date conversion if needed (dict to datetime)
    if 'Date' in df_strat.columns:
        df_strat['Date'] = pd.to_datetime(df_strat['Date'])
    if 'Date' in df_bench.columns:
        df_bench['Date'] = pd.to_datetime(df_bench['Date'])
    
    # --- Results ---
    
    # 1. Performance Overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy CAGR", f"{data['metrics_strategy']['CAGR']:.2%}")
    col2.metric("Max Drawdown", f"{data['metrics_strategy']['Max Drawdown']:.2%}")
    col3.metric("Sharpe Ratio", f"{data['metrics_strategy']['Sharpe Ratio']:.2f}")

    # 2. Equity Curve Comparison
    st.subheader("Performance Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_strat['Date'], y=df_strat['Value'], name="Adaptive Engine", line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=df_bench['Date'], y=df_bench['Value'], name="Benchmark (No Risk Engine)", line=dict(color='gray', dash='dash')))
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Regime & Allocation
    st.subheader("Regime Detection & Asset Allocation")
    
    fig_alloc = px.area(df_strat, x='Date', y=['Equity_Weight', 'Bonds_Weight', 'Cash_Weight'], 
                        title="Dynamic Asset Allocation Over Time",
                        color_discrete_map={'Equity_Weight': 'green', 'Bonds_Weight': 'blue', 'Cash_Weight': 'gray'})
    st.plotly_chart(fig_alloc, use_container_width=True)

    # --- INNOVATION SECTION: Transparent Brain (XAI) ---
    st.markdown("---")
    st.header("🧠 Transparent Brain (Explainable AI)")
    
    if st.session_state.backtester_features is not None:
        from app.core.xai_engine import XAIEngine
        import matplotlib.pyplot as plt
        
        # st.write("Training Surrogate Model...")
        xai = XAIEngine()
        
        # Reconstruct alignment
        feat_df = st.session_state.backtester_features.shift(1).dropna()
        common_dates = df_strat['Date'].tolist()
        feat_df = feat_df[feat_df.index.isin(common_dates)]
        regimes_list = df_strat[df_strat['Date'].isin(feat_df.index)]['Regime'].tolist()
        
        if len(feat_df) > 0:
            xai.train_surrogate(feat_df, regimes_list)
            
            last_date = df_strat['Date'].iloc[-1]
            if last_date in feat_df.index:
                # Use iloc[0:1] to ensure we pass exactly one row as DataFrame
                last_feat_row = feat_df.loc[[last_date]].iloc[0:1]
                fig_shap = xai.get_shap_plot(last_feat_row)
                
                col_xai_1, col_xai_2 = st.columns([2, 1])
                with col_xai_1:
                    st.subheader("Why did the AI choose this Regime?")
                    if fig_shap:
                        st.pyplot(fig_shap)
                with col_xai_2:
                    st.info(f"**Current Regime:** {data['explanation']}")
    else:
        st.warning("Feature data missing for XAI.")

    # --- INNOVATION SECTION: Crisis Lab ---
    st.markdown("---")
    st.header("🧪 Crisis Lab (Interactive Stress Test)")
    
    # Initialize Session State for Crisis (if not exists)
    if 'crisis_simulated' not in st.session_state:
        st.session_state.crisis_simulated = False
        st.session_state.crisis_impact = 0.0
        st.session_state.crisis_msg = ""
    
    with st.expander("Running a Crisis Simulation...", expanded=True):
        shock_val = st.slider("Inject Immediate Market Shock (%)", -20.0, 5.0, -5.0)
        vol_spike = st.slider("Inject Volatility Spike (Multiplier)", 1.0, 5.0, 2.0)
        
        if st.button("Simulate Crisis Reaction"):
            # CURRENT Allocation
            curr_eq = df_strat['Equity_Weight'].iloc[-1]
            curr_cash = df_strat['Cash_Weight'].iloc[-1]
            
            # Impact
            loss = curr_eq * (shock_val / 100.0)
            
            st.session_state.crisis_impact = loss
            st.session_state.crisis_simulated = True
            
            # Response Logic
            if vol_spike > 1.5: 
                st.session_state.crisis_msg = "⚠️ HIGH VOLATILITY DETECTED! Cutting Equity Exposure."
                st.session_state.crisis_new_eq = 0.10 
                st.session_state.crisis_new_cash = 0.80 
            else:
                st.session_state.crisis_msg = "Risk within limits. Holding positions."
                st.session_state.crisis_new_eq = curr_eq
                st.session_state.crisis_new_cash = curr_cash
        
        # Display Results
        if st.session_state.crisis_simulated:
            st.write(f"**Immediate Impact:** Portfolio Value changes by {st.session_state.crisis_impact:.2%}")
            
            st.subheader("AI Engine Reaction:")
            if "Cutting Equity" in st.session_state.crisis_msg:
                st.error(st.session_state.crisis_msg)
                
                curr_eq_display = df_strat['Equity_Weight'].iloc[-1] * 100
                curr_cash_display = df_strat['Cash_Weight'].iloc[-1] * 100
                new_eq_display = st.session_state.crisis_new_eq * 100
                new_cash_display = st.session_state.crisis_new_cash * 100
                
                col_c1, col_c2 = st.columns(2)
                col_c1.metric("Projected Equity Allocation", f"{new_eq_display:.1f}%", f"{new_eq_display - curr_eq_display:.1f}%")
                col_c2.metric("Projected Cash Allocation", f"{new_cash_display:.1f}%", f"{new_cash_display - curr_cash_display:.1f}%")
            else:
                st.success(st.session_state.crisis_msg)

else:
    st.info("In the sidebar, select a ticker and date range, then click 'Run Simulation'.")
