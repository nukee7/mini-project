import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np

class IntegratedGradientsXAI:
    """
    Advanced Explainable AI (XAI) module using Integrated Gradients.
    Designed for interpreting model decisions in medical settings 
    to separate true clinical biomarkers from 'spurious' hospital noise.
    """
    
    def __init__(self, model: nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps
        
    def generate_baselines(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Creates a baseline input matrix (e.g., zero tensor for images).
        Can be customized for specific tabular or image medical data.
        """
        return torch.zeros_like(inputs)
        
    def attribute(self, inputs: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Calculates Integrated Gradients for the given inputs and target.
        IG(x) = (x - x') * int_{\alpha=0}^1 grad(F(x' + \alpha(x - x'))) d\alpha
        """
        self.model.eval()
        baseline = self.generate_baselines(inputs)
        
        # 1. Scale inputs and compute interpolations
        alphas = torch.linspace(0.0, 1.0, steps=self.steps).tolist()
        scaled_features = [
            baseline + alpha * (inputs - baseline) 
            for alpha in alphas
        ]
        
        # 2. Compute gradients for all scaled inputs
        gradients = []
        for scaled_input in scaled_features:
            scaled_input.requires_grad_(True)
            output = self.model(scaled_input)
            
            # Select the output for the target class
            class_score = output[0, target_class]
            
            self.model.zero_grad()
            class_score.backward()
            gradients.append(scaled_input.grad.detach().numpy())
            
        # 3. Approximate the integral via Riemann sum
        avg_gradients = np.average(gradients, axis=0)
        
        # 4. Multiply with (input - baseline)
        integrated_gradients = (inputs.detach().numpy() - baseline.detach().numpy()) * avg_gradients
        
        return integrated_gradients
        
    def federated_aggregate_explanations(self, local_attributions: List[np.ndarray]) -> np.ndarray:
        """
        Aggregates local feature attributions cross-silo to construct
        a global fidelity map, ensuring explanations generalize across
        demographics rather than over-indexing on non-IID local noise.
        """
        # Simple mean aggregation over attributions. 
        # For production, this could incorporate Fairness-q weighting as well.
        return np.mean(local_attributions, axis=0)
