import torch
from captum.attr import IntegratedGradients

class Explainer:
    """Implement Integrated Gradients locally or globally to define model attributions."""
    def __init__(self, model):
        self.ig = IntegratedGradients(model)
        
    def extract_attributions(self, input_tensor: torch.Tensor, target: int = 1):
        attributions = self.ig.attribute(input_tensor, target=target)
        return attributions

class FairnessAuditor:
    """
    Generates Fairness Audits guaranteeing that high-risk groups aren't underserved.
    Calculates dynamic False Negative Rates (FNR) mapping missing interventions.
    """
    @staticmethod
    def generate_fnr_report(hospital_stats: dict):
        """
        hospital_stats: maps string hospital_names to their confusion matrix components:
        format { "Cleveland": {"FN": 12, "TP": 140}, ... }
        """
        print("--- EQUITABLE ETHICS: FAIRNESS AUDITOR ---")
        highest_fnr = 0.0
        underserved_group = None
        
        for client, stats in hospital_stats.items():
            fn = stats.get("FN", 0)
            tp = stats.get("TP", 1) # Prevent div 0
            
            # FNR = FN / (FN + TP) represents patients missing critical disease classification
            fnr = fn / (fn + tp)
            print(f"Hospital {client} False Negative Rate (FNR): {fnr:.4f}")
            
            if fnr > highest_fnr:
                highest_fnr = fnr
                underserved_group = client
                
        print("========================================")
        print(f"WARNING THRESHOLD - Underserved Client: {underserved_group} with FNR {highest_fnr:.4f}")
        return underserved_group
