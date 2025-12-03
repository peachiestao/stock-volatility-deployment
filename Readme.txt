Project Overview

This project implements a complete machine-learning workflow for financial volatility prediction, including:

1. Data preprocessing and model training (build_model.ipynb)

2. Model deployment and inference pipeline (deploy.py)

3. Automated data drift monitoring (monitor.py)

4. Automated model retraining when drift exceeds a defined threshold (train_model.py)

The goal is to maintain a fully operational MLOps lifecycle where the model adapts to changing data distributions.

Repository Structure:
.
├── build_model.ipynb       # Data Preprocessing + model training + model saving script
├── deploy.py               # Model deployment and prediction service
├── monitor.py              # Data drift monitoring pipeline
├── train_model.py          # Automated retraining pipeline (triggered by monitor)
├── reference_data.csv      # Data used to define baseline distributions
├── production_logs.csv     # New (live) data used for monitoring
└── monitor.log             # monitoring script log
└── system.log              # deploy script log
├── best_model/
│   └── final_model.keras   # Saved best model
    └── scaler.pkl
├── reports/
│   ├── tsla_featuredrift_report.html  # feature drift report
│   ├── tsla_targetdrift_report.html   # data drift report
│   └── tsla_datadrift_report.json
└── requirements.txt  # installation packages





