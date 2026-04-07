import flwr as fl
from src.federated.server import HeterogeneousFedProx
from src.federated.client import get_client_fn

def main():
    """
    Entry mechanism mapped precisely to deploy the High-Performance 
    Federated Architecture utilizing Opacus, Captum, and Flower.
    """
    print("Initiating Reproducible High-Risk Disease Prediction Simulation...")
    print("WARNING: Run /scripts/setup_data.py first to partition local UCI EHR records.")
    
    # Predefined international nodes targeting UCI partitions
    hospital_nodes = ["Cleveland", "Hungary", "Switzerland", "Long_Beach_VA"]
    
    strategy = HeterogeneousFedProx(
        proximal_mu=0.5,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(hospital_nodes),
        min_evaluate_clients=len(hospital_nodes),
        min_available_clients=len(hospital_nodes)
    )
    
    print(f"Server instantiated with Heterogeneous FedProx strategy over {len(hospital_nodes)} hospitals.")
    
    fl.simulation.start_simulation(
        client_fn=get_client_fn(hospital_nodes),
        num_clients=len(hospital_nodes),
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
