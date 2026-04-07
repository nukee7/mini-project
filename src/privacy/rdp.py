import math
import numpy as np
from typing import List, Tuple

class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant for Federated Learning.
    Calculates the accumulated privacy budget (epsilon, delta) over T rounds
    given a noise multiplier and subsampling rate.
    """
    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.rdp_history = []
        self.orders = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]

    def _compute_rdp_sgm(self, q: float, noise_multiplier: float, alpha: float) -> float:
        """
        Computes RDP of the Subsampled Gaussian Mechanism for a single round.
        Approximation for small q. 
        """
        if q == 0:
            return 0.0
        if noise_multiplier == 0:
            return float('inf')
        
        # Using standard Gaussian Mechanism RDP for alpha > 1
        return (q ** 2 * alpha) / (2 * (noise_multiplier ** 2))

    def step(self, q: float, noise_multiplier: float):
        """Record a single federated learning round's RDP contribution."""
        rdp_round = [self._compute_rdp_sgm(q, noise_multiplier, alpha) for alpha in self.orders]
        self.rdp_history.append(rdp_round)

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Convert cumulative RDP array to standard (epsilon, delta)-DP.
        Returns the optimal epsilon and the configured target delta.
        """
        if not self.rdp_history:
            return 0.0, self.target_delta
            
        cumulative_rdp = np.sum(self.rdp_history, axis=0)
        
        epsilons = []
        for i, alpha in enumerate(self.orders):
            eps = cumulative_rdp[i] + (math.log(1 / self.target_delta) / (alpha - 1))
            epsilons.append(eps)
            
        best_epsilon = min(epsilons)
        return best_epsilon, self.target_delta

    def inject_noise(self, update: np.ndarray, noise_multiplier: float, sensitivity: float = 1.0) -> np.ndarray:
        """
        Inject Gaussian noise into the parameter update for DP assurance.
        """
        scale = noise_multiplier * sensitivity
        noise = np.random.normal(0, scale, size=update.shape)
        return update + noise
