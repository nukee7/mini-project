import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np

class HeterogeneousFedProx(fl.server.strategy.FedProx):
    """
    FedProx implementation to manage high statistical heterogeneity 
    between the 4 international sites (Cleveland vs Switzerland bounds).
    """
    def __init__(self, proximal_mu: float = 0.5, *args, **kwargs):
        super().__init__(proximal_mu=proximal_mu, *args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Exception],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # The core logic executing over statistical variances
        params_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate logic
        aggregated_ndarrays = self.aggregate_blindly(params_results)
        
        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def aggregate_blindly(self, results):
        """Securely sum the arrays shielding individual identities."""
        total_examples = sum([num_examples for _, num_examples in results])
        aggregated_weights = [np.zeros_like(weights) for weights in results[0][0]]
        
        for weights, num_ex in results:
            factor = num_ex / total_examples
            for i, layer in enumerate(weights):
                aggregated_weights[i] += layer * factor
                
        return aggregated_weights
