import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import shap

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components import render_sidebar, plot_holographic_risk, render_metrics
from core.data_loader import fetch_data
from core.backtester import Backtester
from core.stress_tester import inject_shock

st.set_page_config(page_title="EchoRegime Dashboard", layout="wide")

st.title("EchoRegime: Autonomous Adaptive Portfolio & Risk Engine")

inputs = render_sidebar()

if st.button("Run Simulation"):
    with st.spinner("Fetching data and running EchoRegime simulation..."):
        # 1. Fetch Data
        data = fetch_data(inputs["tickers"], inputs["start_date"], inputs["end_date"])
        
        if data.empty:
            st.error("No data fetched. Please check tickers and date range.")
        elif len(data) < 252:
            st.error("Insufficient data for simulation. Please select a date range longer than 1 year (252 trading days).")
        else:
            # 2. Run Backtest (Risk ON vs Risk OFF)
            backtester = Backtester(data, initial_capital=inputs["initial_capital"])
            
            # Risk ON
            results_risk, weights_risk = backtester.run_backtest(use_risk_engine=True)
            metrics_risk = backtester.calculate_metrics(results_risk['Portfolio Value'])
            latest_shap = backtester.latest_shap_values
            latest_features = backtester.latest_features
            
            # Risk OFF
            results_no_risk, _ = backtester.run_backtest(use_risk_engine=False)
            metrics_no_risk = backtester.calculate_metrics(results_no_risk['Portfolio Value'])
            
            # 3. Visualization
            
            # Metrics Comparison
            st.header("Performance Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ With Risk Engine")
                st.json(metrics_risk)
            with col2:
                st.subheader("❌ Without Risk Engine")
                st.json(metrics_no_risk)
            
            # Portfolio Value Comparison
            st.header("Portfolio Performance (Risk Engine vs Buy-and-Hold)")
            combined_df = pd.DataFrame({
                "With Risk Engine": results_risk['Portfolio Value'],
                "Without Risk Engine": results_no_risk['Portfolio Value']
            })
            st.line_chart(combined_df)
            
            # Allocation
            st.header("Allocation History (Adaptive)")
            st.area_chart(weights_risk)
            
            # Explainability (SHAP)
            st.header("🧠 Explainability (Why did the AI choose this regime?)")
            if latest_shap is not None and latest_features is not None:
                st.write(f"Explanation for the latest decision on {data.index[-1].date()}")
                
                # SHAP expects 2D array, latest_shap might be list of arrays for each class
                # Random Forest binary/multiclass returns list of arrays
                # Let's take the SHAP values for the predicted class or just class 1 (High Vol)
                
                # Check structure
                if isinstance(latest_shap, list):
                    # Multiclass: [n_samples, n_features] for each class
                    # We visualize impact on Class 1 (High Vol) or Class 0 (Low Vol)
                    shap_values_to_plot = latest_shap[1] # Class 1 (Bear/High Vol) usually interesting
                    # If model only has 2 classes, sometimes it returns just 1 array? 
                    # shap.TreeExplainer for RF usually returns list.
                else:
                    shap_values_to_plot = latest_shap

                try:
                    # Summary plot (bar) for local explanation
                    # shap.force_plot is interactiveJS, might be hard in Streamlit without component
                    # shap.waterfall_plot is good for single prediction
                    
                    fig, ax = plt.subplots()
                    # We use summary_plot of just this single instance
                    shap.summary_plot(shap_values_to_plot, latest_features, plot_type="bar", show=False)
                    st.pyplot(fig)
                    st.caption("Feature contribution to 'High Volatility' regime probability.")
                except Exception as e:
                    st.warning(f"Could not render SHAP plot: {e}")
            else:
                st.info("No SHAP values available (simulation too short?).")

            # Stress Testing
            st.header("🔥 Stress Testing")
            if st.button("Inject -20% Market Crash"):
                with st.spinner("Running Stress Test..."):
                    shocked_data = inject_shock(data, drop_pct=0.20)
                    
                    # Run on shocked
                    bt_stress = Backtester(shocked_data, inputs["initial_capital"])
                    res_stress, w_stress = bt_stress.run_backtest(use_risk_engine=True)
                    res_stress_no_risk, _ = bt_stress.run_backtest(use_risk_engine=False)
                    
                    st.image("https://media.giphy.com/media/constant-panic/giphy.gif", caption="Stress Test Running...", width=200) # Optional fun
                    
                    # Plot comparison
                    stress_df = pd.DataFrame({
                        "Protected (Risk Engine)": res_stress['Portfolio Value'],
                        "Unprotected": res_stress_no_risk['Portfolio Value']
                    })
                    st.line_chart(stress_df)
                    
                    # Calculate drop
                    final_protected = res_stress['Portfolio Value'].iloc[-1]
                    final_unprotected = res_stress_no_risk['Portfolio Value'].iloc[-1]
                    
                    st.metric("Capital Preserved (Protected)", f"${final_protected:,.2f}")
                    st.metric("Capital Remaining (Unprotected)", f"${final_unprotected:,.2f}", delta=f"{final_protected - final_unprotected:,.2f} preserved")

    
