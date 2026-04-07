import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from collections import OrderedDict
from pathlib import Path

# Adjust paths based on deployment module
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.model import DiseaseNet, PrivacyWrapper

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

class SecureHospitalClient(fl.client.NumPyClient):
    """
    Skeletal implementation of the local Federated Node.
    Every node intrinsically enforces Differential Privacy on its tabular data
    via Meta's Opacus Engine.
    """
    def __init__(self, hospital_id: str):
        self.hospital_id = hospital_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiseaseNet(input_dim=13).to(self.device)
        self.load_hospital_ehr()

    def load_hospital_ehr(self):
        """Loads heavily-imputed local tabular records."""
        file_path = DATA_DIR / f"{self.hospital_id}.csv"
        # Wait until download module provides datasets
        if not file_path.exists():
            print(f"Data for {self.hospital_id} not found. Run scripts/setup_data.py first.")
            return

        df = pd.read_csv(file_path)
        X = df.drop("target", axis=1).values
        y = df["target"].values
        
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.long)
        )
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        self.trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.valloader = DataLoader(val_dataset, batch_size=32)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Standard Opacus-enabled training."""
        if not hasattr(self, 'trainloader'):
            return self.get_parameters(config), 0, {}

        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        # Inject Differential Privacy explicitly into the PyTorch engine
        self.model.train()
        private_model, private_optimizer, private_dataloader, privacy_engine = PrivacyWrapper.make_private(
            model=self.model,
            optimizer=optimizer,
            data_loader=self.trainloader,
            target_epsilon=3.0,
            epochs=1
        )
        
        for inputs, labels in private_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            private_optimizer.zero_grad()
            outputs = private_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            private_optimizer.step()
            
        return self.get_parameters(config), len(private_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        if not hasattr(self, 'valloader'):
            return 0.0, 0, {}

        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        loss, correct, total = 0.0, 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return float(loss / len(self.valloader)), total, {"accuracy": float(correct / total)}

def get_client_fn(hospital_id_map: list):
    """Maps numerical internal Node IDs cleanly to geographical names."""
    def client_fn(cid: str) -> fl.client.Client:
        hospital_name = hospital_id_map[int(cid)]
        return SecureHospitalClient(hospital_name).to_client()
    return client_fn
