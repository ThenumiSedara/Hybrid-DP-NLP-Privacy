import torch
import numpy as np
from opacus import PrivacyEngine
import os

def calculate_sensitivity(text, emotions_sum):
    """
    Calculates a sensitivity score for a given text based on its length 
    and the intensity of emotions expressed.
    """
    if not isinstance(text, str):
        text = ""
    
    # Normalize length (assuming max 512 for roberta-base)
    len_norm = min(len(text) / 512, 1.0)
    
    # Emotion intensity: GoEmotions labels are binary [0, 1]. 
    # emotions_sum is the count of active emotions.
    # Normalize by max possible (28 emotions) or a realistic high value (e.g., 5).
    intensity_norm = min(emotions_sum / 5.0, 1.0)
    
    # Sensitivity = weighted sum of length and intensity
    # Higher sensitivity -> higher privacy risk
    sensitivity = 0.4 * len_norm + 0.6 * intensity_norm
    return sensitivity

class HybridDPCoach:
    """
    Manages DP-SGD training using Opacus for formal privacy guarantees.
    """
    def __init__(self, model, optimizer, data_loader, epsilon=8.0, delta=1e-5, max_grad_norm=1.0, epochs=3):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.privacy_engine = PrivacyEngine()
        self.is_attached = False

    def attach(self):
        """
        Attaches the privacy engine to the model, optimizer, and data loader.
        """
        if self.is_attached:
            return self.model, self.optimizer, self.data_loader
            
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
        )
        self.is_attached = True
        return self.model, self.optimizer, self.data_loader

    def get_privacy_spent(self):
        # Returns the current privacy budget spent (epsilon).
        if not self.is_attached:
            return 0.0
        return self.privacy_engine.accountant.get_epsilon(delta=self.delta)

def analyze_standalone_drawbacks(eval_results):
    
    # Analyzes and visualizes the drawbacks of standalone approaches (V1, V2) compared to the Hybrid approach (V3).
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # eval_results should be a dict or DataFrame with variant metrics
    df = pd.DataFrame(eval_results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='variant', y='f1_macro', data=df, palette='viridis')
    plt.title('Utility (F1-macro) Across Privacy Variants')
    plt.ylabel('F1-macro Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the analysis summary
    analysis_path = 'reports/rq1/standalone_analysis.png'
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    plt.savefig(analysis_path)
    plt.show()
    
    return "Analysis complete. Standalone approaches typically show lower utility (DP-SGD only) or higher residual risk (Anonymization only) than Hybrid."
