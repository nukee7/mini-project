import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np

class FairnessAwareFedProx(fl.server.strategy.FedProx):
    """
    Advanced FedProx aggregator with Fairness-Aware Client Drift reweighting.
    Mitigates minority performance degradation by boosting update weights 
    for clients experiencing higher local loss.
    """
    
    def __init__(self, 
                 proximal_mu: float = 0.1, 
                 fairness_q: float = 1.5,
                 *args, **kwargs):
        super().__init__(proximal_mu=proximal_mu, *args, **kwargs)
        self.fairness_q = fairness_q

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # Convert results parameters to NumPy arrays
        client_updates = []
        client_losses = []
        total_examples = sum([fit_res.num_examples for _, fit_res in results])
        
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            loss = fit_res.metrics.get("loss", 1.0)
            client_losses.append(loss)
            client_updates.append((weights, fit_res.num_examples, loss))
            
        adjusted_weights = []
        for loss, num_ex in zip(client_losses, [c[1] for c in client_updates]):
            fairness_multiplier = loss ** self.fairness_q
            adjusted_weights.append((num_ex / total_examples) * fairness_multiplier)
            
        weight_sum = sum(adjusted_weights)
        norm_weights = [w / weight_sum for w in adjusted_weights]
        
        aggregated_ndarrays = self._aggregate_with_weights(client_updates, norm_weights)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if results and results[0][1].metrics:
            for metric in results[0][1].metrics.keys():
                metric_vals = [res.metrics[metric] for _, res in results if metric in res.metrics]
                if metric_vals:
                    metrics_aggregated[metric] = sum(metric_vals) / len(metric_vals)
                    
        return parameters_aggregated, metrics_aggregated

    def _aggregate_with_weights(self, client_updates, norm_weights):
        """Performs weighted averaging of model updates."""
        if len(client_updates[0][0]) == 0:
            return []
            
        aggregated_weights = [np.zeros_like(weights) for weights in client_updates[0][0]]
        for (weights, _, _), weight in zip(client_updates, norm_weights):
            for i, layer in enumerate(weights):
                aggregated_weights[i] += layer * weight
        return aggregated_weights

def get_strategy() -> fl.server.strategy.Strategy:
    """Returns the aggregation strategy for the server."""
    strategy = FairnessAwareFedProx(
        proximal_mu=0.1,
        fairness_q=1.5,
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2, # For quick testing, reduced from 10 to 2
    )
    return strategy
