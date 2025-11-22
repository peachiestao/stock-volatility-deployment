# monitor_drift.py
import pandas as pd
import json
import os

from evidently import Report
from evidently.presets import DataDriftPreset


# === CONFIGURATION ===
REFERENCE_FILE = 'reference_data.csv'
PRODUCTION_FILE = 'production_logs.csv'
REPORT_OUTPUT = 'monitoring_report.html'
REPORT_JSON= 'monitoring_report.json'

DRIFT_THRESHOLD = 0.3  # If more than 50% of features drift, trigger alert

def trigger_retraining():
    """
    Mechanism to trigger action. 
    In a real system, this might call an Airflow DAG or AWS Lambda.
    """
    print("\n[ACTION TRIGGERED]  MODEL RETRAINING REQUESTED")
    print("Reason: Significant Data Drift detected.")
    # Example: os.system("python train_model.py")

def send_alert(drift_score, drifted_features):
    """
    Simulate an alert system (Slack/Email/PagerDuty).
    """
    print(f"\n[ALERT]  Data Drift Detected!")
    print(f"Drift Score: {drift_score:.2f}")
    print(f"Drifting Features: {drifted_features}")

def run_monitoring():
    # 1. Load Data
    if not os.path.exists(PRODUCTION_FILE):
        print("No production logs found yet.")
        return

    ref_data = pd.read_csv(REFERENCE_FILE)
    prod_data = pd.read_csv(PRODUCTION_FILE)

    # Drop timestamp/score for feature comparison, keep Prediction_Class if available in ref
    # For this example, we focus on Input Feature Drift
    features = [col for col in ref_data.columns if col != 'Target'] 
    
    # Align columns (Prod data has extra log columns)
    current_data = prod_data[features]

    print(f"Analyzing {len(current_data)} production samples against baseline...")

    # 2. Create Evidently Report
    report = Report(metrics=[
        DataDriftPreset(),  # Checks inputs
        # TargetDriftPreset() # Checks if output distribution changed (requires target in ref)
    ])

    eval=report.run(reference_data=ref_data, current_data=current_data)

    # 3. Save Visual Report
    
    eval.save_html(REPORT_OUTPUT)
    eval.save_json(REPORT_JSON)
    print(f"Report saved to {REPORT_OUTPUT}")

    # 4. Check for Anomalies programmatically
    # Extract metric result as a Python dict
    results = eval.dict()
    
    # Navigate JSON structure to find drift share
    DRIFT_THRESHOLD = 0.3   # you choose

    # --- Extract global drift share from DriftedColumnsCount ---
    drift_count_metric = results["metrics"][0]   # ALWAYS DriftedColumnsCount

    drift_share = drift_count_metric["value"]["share"]
    num_drifted = drift_count_metric["value"]["count"]

    print(f"Drift Share: {drift_share}, Drifted Features: {num_drifted}")

    # --- Extract per-feature drift based on ValueDrift metrics ---
    drifted_columns = []

    for metric in results["metrics"]:
        if metric["config"]["type"] == "evidently:metric_v2:ValueDrift":
            col = metric["config"]["column"]
            score = metric["value"]
            threshold = metric["config"]["threshold"]

            if score > threshold:
                drifted_columns.append(col)

    # --- Decision Logic ---
    if drift_share > DRIFT_THRESHOLD:
        print("[ALERT] Significant drift detected!")
        print("Drifted columns:", drifted_columns)
        send_alert(drift_share, drifted_columns)
        trigger_retraining()
    else:
        print("[STATUS] Healthy — no significant drift detected.")


if __name__ == "__main__":
    run_monitoring()