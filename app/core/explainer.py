import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Explainer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def explain(self, regime, allocation, metrics):
        """
        Generates a natural language explanation of the portfolio's state.
        """
        # Context for the LLM
        prompt = f"""
        You are a senior portfolio manager. Explain the current market situation and portfolio decision.
        
        **Data:**
        - Detected Regime: {regime}
        - Portfolio Allocation: Equity {allocation.get('Equity', 0):.1%}, Bonds {allocation.get('Bonds', 0):.1%}, Cash {allocation.get('Cash', 0):.1%}
        - Performance Metrics: CAGR {metrics.get('CAGR', 0):.1%}, Max Drawdown {metrics.get('Max Drawdown', 0):.1%}
        
        **Task:**
        Provide a concise, professional explanation of why this allocation is appropriate for the current regime. 
        Focus on risk management logic. max 3 sentences.
        """
        
        if self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama-3.1-8b-instant",
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                return f"GenAI Connection Error: {e}. Falling back to template."

        # Template-based fallback
        explanation = f"""
        **Market Regime Detected:** {regime}
        
        **Action Taken:**
        The system has adjusted the portfolio allocation to:
        - Equity: {allocation.get('Equity', 0):.1%}
        - Bonds: {allocation.get('Bonds', 0):.1%}
        - Cash: {allocation.get('Cash', 0):.1%}
        
        **Reasoning:**
        """
        
        if regime == 'Crash':
            explanation += "Extreme market stress detected. Maximizing capital preservation (High Cash/Gold)."
        elif regime == 'High Volatility':
            explanation += "Volatility is elevated. Reducing risk assets to protect against potential downside."
        elif regime == 'Bearish':
            explanation += "Trend is negative (Price < 200 SMA). Defensive positioning favored."
        else:
            explanation += "Market conditions are favorable (Calm & Trending Up). Increasing equity exposure to capture upsides."
            
        return explanation
