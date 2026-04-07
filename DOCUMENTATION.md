# Detailed Academic Documentation
**Explainable and Ethical Federated AI for High-Risk Disease Prediction**

This documentation provides the theoretical mechanisms and academic justifications behind our robust Plug-and-Play research architecture. It serves to outline how we conquer massive statistical heterogeneity, demographic unfairness, and severe clinical missingness in cross-silo healthcare settings.

## 🌟 Core Features & Academic Innovations

### 1. **Ethical Data Partitioning (UCI 4-Database Strategy)**
- **What it does**: Pulls down the native UCI Heart Disease datasets spanning four distinct international hospitals: **Cleveland, Hungary, Switzerland, and Long Beach VA**. 
- **The Innovation**: The Swiss and Long Beach geographical nodes are notoriously sparse mathematically. Rather than dropping missing values (which disproportionately Discriminates against target demographics and introduces severe bias against underserved regions), we employ autonomous **Iterative Imputation** per node. It recursively models and predicts the minority records securely before Federated aggregation begins.

### 2. **Privacy-Preserving Federated Architecture**
- **What it does**: We decouple the hospitals to run their own localized ML training locally, connected via a standard `flwr` (Flower) server.
- **The Innovation**: 
  - **FedProx & Secure Blind Aggregation**: We conquer extreme regional Non-IID variance natively via FedProx's proximal term. Furthermore, the server only ever computes blinded weighted averages, guaranteeing it cannot reverse-engineer a specific client's weights natively.
  - **Opacus Differential Privacy**: Every local PyTorch `DiseaseNet` is mathematically bounded by Meta's Opacus Engine. It aggressively clips layer gradients to enforce a strict Privacy Budget constraint mathematically proven to scale $\epsilon \le 3.0$.

### 3. **The "Ethics & XAI" Suite**
- **What it does**: Proves that the "Black Box" nature of Neural Networks isn't hiding systemic discrimination against minority groups.
- **The Innovation**: 
  - **Captum Integrated Gradients**: Replaces basic LIME algorithms with global Integrated Gradients targeting high-risk features, resulting in direct clinical biomarker attribution maps (saved via Matplotlib configurations).
  - **Fairness Auditor**: Implements Equality of Opportunity (TPR) and Predictive Equality (FPR) constraints, algorithmically verifying whether the model's False Negative Rates are equitably balanced among all 4 international silos.

---

## 📁 Repository Structure

```tree
.
├── configs/
│   └── config_demo.yaml            # Adjust simulation rounds, epsilon limits, and strategies
├── data/                           # (Locally generated via script) Raw and partitioned EHR data
├── results/                        # Auto-generated destination for XAI plots and Privacy curves
├── scripts/
│   └── setup_data.py               # The automated Iterative Imputer and target downloaded
├── src/
│   ├── federated/
│   │   ├── client.py               # Localized Opacus training nodes wrapped via flwr 
│   │   ├── server.py               # HeterogeneousFedProx Secure aggregator
│   │   ├── explainability.py       # Hooks for Fairness Auditor and Integrated Gradients
│   │   └── advanced_metrics.py     # Suite that visually plots Opacus bounding parameters
│   └── models/
│       └── model.py                # Core PyTorch architecture customized for 13 features
├── main.py                         # Internal Server instantiation 
└── run_experiment.py               # 🚀 Master Entry Point
```
