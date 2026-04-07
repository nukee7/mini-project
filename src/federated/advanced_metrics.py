import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from captum.attr import IntegratedGradients

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class PublicationSuite:
    """
    Core engine handling advanced visual analysis for the Research Paper Suite.
    Integrates XAI visual mapping, Privacy Budget analysis, and Model Fairness.
    """
    
    @staticmethod
    def plot_privacy_budget(epsilons: list, deltas: list, rounds: int):
        """Generates rigorous Privacy Budget curve (Epsilon vs Round)."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, rounds + 1), epsilons, marker='o', linewidth=2, color='#2c3e50')
        plt.title('Strict Differential Privacy Constraints over Federated Rounds')
        plt.xlabel('Federation Round')
        plt.ylabel(r'Privacy Budget Limit ($\epsilon$)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=3.0, color='r', linestyle='dashdot', label=r'Regulatory Constraint ($\epsilon \leq 3.0$)')
        plt.legend()
        plt.savefig(RESULTS_DIR / 'privacy_budget_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def generate_fairness_audit(client_confusion_matrices: dict):
        """
        Proof of 'Ethical constraint' metrics.
        Computes Equality of Opportunity (TPR/FNR limits) & Predictive Equality (FPR).
        """
        print("--- DEMOGRAPHIC ETHICS AUDIT ---")
        
        # Calculate FPR (False Positive Rate) & FNR (False Negative Rate)
        results = {}
        for hospital, matrix in client_confusion_matrices.items():
            fn = matrix.get("FN", 0)
            tp = matrix.get("TP", 1)
            fp = matrix.get("FP", 0)
            tn = matrix.get("TN", 1)
            
            tpr_equality = tp / (tp + fn) # Equality of Opportunity
            fpr_predictive = fp / (fp + tn) # Predictive Equality
            
            results[hospital] = {
                "Equality of Opportunity (TPR)": tpr_equality,
                "Predictive Equality (FPR)": fpr_predictive
            }
            
            print(f"Hospital [{hospital}] -> Eq. Opp. (TPR): {tpr_equality:.3f} | Pred. Eq. (FPR): {fpr_predictive:.3f}")
            
        return results

    @staticmethod
    def plot_global_integrated_gradients(model, input_tensors, feature_names):
        """Extracts XAI logic producing global standard attribution maps."""
        ig = IntegratedGradients(model)
        
        # Calculate absolute mean attributions bridging across all inferences
        attributions = ig.attribute(input_tensors, target=1)
        mean_attrs = attributions.mean(dim=0).detach().cpu().numpy()
        
        # Visualize Feature Attributions for publication standard
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, np.abs(mean_attrs), align='center', color='#3498db')
        plt.xlabel('Absolute Mean Attribution Score ($Integrated$ $Gradients$)')
        plt.title('Global Feature Importance against High-Risk Diagnosis')
        plt.gca().invert_yaxis()
        
        plt.savefig(RESULTS_DIR / 'integrated_gradients_xai.png', dpi=300, bbox_inches='tight')
        plt.close()
