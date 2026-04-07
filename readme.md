# Explainable and Ethical Federated AI for High-Risk Disease Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Federated Framework](https://img.shields.io/badge/flower-1.0+-green.svg)](https://flower.dev/)
[![Differential Privacy](https://img.shields.io/badge/opacus-enabled-purple.svg)](https://opacus.ai/)
[![XAI](https://img.shields.io/badge/captum-integrated-orange.svg)](https://captum.ai/)

This repository provides a 100% plug-and-play, publication-ready algorithmic suite for high-risk demographic medical prediction. 

📖 **For deep academic specifications, theoretical innovations (Opacus bounds, Fairness Auditing, Iterative Imputation), and advanced directory mapping, please refer to [DOCUMENTATION.md](DOCUMENTATION.md).**

---

## 🚀 How to Use This Framework

This architecture is deliberately decoupled into specific commands. All prerequisite dependencies operate flawlessly in Linux/VS Code and natively execute "out-of-the-box" on Google Colab ecosystems.

### **Step 1: Environment Setup**
Ensure your local environment possesses standard Scientific Python ML bindings:
```bash
pip install "flwr[simulation]" torch torchvision numpy pandas scikit-learn opacus captum pyyaml
```

### **Step 2: Automated Dataset Preparation**
Execute the data script. This connects securely to the UCI API architecture, downloads the `.data` endpoints, and creates your local heavily-imputed `data/raw` and `data/partitions` structures via Scikit-Learn.
```bash
python scripts/setup_data.py
```
*Note: Any output data is safely handled natively and ignored by git versions as we prioritize HIPAA-adjacent security.*

### **Step 3: Trigger the FL Experiment**
Once the `/data/partitions/*.csv` files exist locally, execute the Master orchestrator:
```bash
python run_experiment.py
```
**This script automatically triggers:**
1. A multi-node FL simulated environment scaling across `num_rounds`.
2. Evaluates exactly against the bounds stated in `configs/config_demo.yaml`.
3. Instantiates the `advanced_metrics.py` hook which creates and saves statistical figures right into your `/results` directory. 

### **Step 4: Audit & Tweak**
Open `configs/config_demo.yaml` to dynamically adjust the system payload! Try reducing the `privacy_engine.target_epsilon` to `<1.0` and observe exactly how the models dynamically enforce tighter gradient variance inside standard Opacus boundaries!