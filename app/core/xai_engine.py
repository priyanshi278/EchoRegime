import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class XAIEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.explainer = None
        self.feature_names = None

    def train_surrogate(self, features_df, regimes_list):
        """
        Trains a surrogate Random Forest model to mimic the Rule-Based Regime Detector.
        This allows us to use SHAP to explain the rules.
        """
        # Clean data
        X = features_df.copy().dropna()
        y = regimes_list[-len(X):] # Align lengths
        
        # Fit model
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        
        # Initialize SHAP explainer
        # TreeExplainer is fast for Trees
        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_plot(self, current_features_row):
        """
        Generates a SHAP force plot for a single prediction.
        Returns a matplotlib figure.
        """
        if self.explainer is None:
            return None
            
        try:
            # Calculate SHAP values for this instance
            shap_values = self.explainer.shap_values(current_features_row)
            
            # Identify the predicted class
            prediction = self.model.predict(current_features_row)[0]
            try:
                class_index = list(self.model.classes_).index(prediction)
            except (ValueError, IndexError):
                class_index = 0
            
            # Robustly handle SHAP output structure
            if isinstance(shap_values, list):
                if class_index < len(shap_values):
                    shap_val_class = shap_values[class_index]
                else:
                    shap_val_class = shap_values[-1]
            else:
                shap_val_class = shap_values
                
            # Extract values
            # Ensure vals is a clean 1D array of FLOATS
            vals = np.array(shap_val_class)
            vals = vals.reshape(-1) # Force flat
            try:
                vals = vals.astype(float)
            except Exception:
                print(f"Warning: SHAP vals could not be cast to float. Shape: {vals.shape}, Dtype: {vals.dtype}")
                vals = np.zeros(len(self.feature_names))
                
            feature_names = self.feature_names
            
            # Debug Mismatch
            if len(vals) != len(feature_names):
                # Truncate vals if too long
                if len(vals) > len(feature_names):
                    vals = vals[:len(feature_names)]
                # If vals too short, we pad with zeros?
                elif len(vals) < len(feature_names):
                    vals = np.pad(vals, (0, len(feature_names) - len(vals)), 'constant')

            # Sort by magnitude
            indices = np.argsort(np.abs(vals))
            
            # Filter indices to ensure they are valid for feature_names (double safety)
            valid_indices = [i for i in indices if i < len(feature_names)]
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Safe color logic
            colors = ['red' if x < 0 else 'green' for x in vals[valid_indices]]
            
            ax.barh(range(len(valid_indices)), vals[valid_indices], color=colors)
            ax.set_yticks(range(len(valid_indices)))
            ax.set_yticklabels([feature_names[i] for i in valid_indices])
            ax.set_xlabel(f"Contribution to '{prediction}' Regime")
            ax.set_title("Why did the AI choose this Regime? (SHAP Values)")
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"XAI Engine Error: {e}")
            import traceback
            traceback.print_exc()
            return None
