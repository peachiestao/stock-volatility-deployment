# monitor_drift.py
import pandas as pd
import json
import os
import subprocess
import logging

from evidently import Report
from evidently.presets import DataDriftPreset

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# === CONFIGURATION ===
REFERENCE_FILE = 'reference_data.csv'
PRODUCTION_FILE = 'production_logs.csv'
REPORT_OUTPUT = 'monitoring_report.html'
REPORT_JSON = 'monitoring_report.json'

DRIFT_THRESHOLD = 0.3  # If more than 30% drift → alert


def trigger_retraining():
    """Trigger External Retraining."""
    logger.warning("ACTION TRIGGERED: Model retraining requested due to drift.")
    print("\n[ACTION TRIGGERED] MODEL RETRAINING REQUESTED")
    print("Reason: Significant Data Drift detected.")

    try:
        logger.info("Launching local retraining job using train_model.py...")
        subprocess.run(["python", "train_model.py"], check=True)
        logger.info("Retraining completed successfully.")
        print("[SUCCESS] Retraining completed. New model ready.")
    except Exception as e:
        logger.exception("Retraining failed.")
        print("[ERROR] Retraining failed:", str(e))


def send_alert(drift_score, drifted_features):
    """Simulate an alert system."""
    logger.error(f"DATA DRIFT ALERT — Score: {drift_score:.2f}, Features: {drifted_features}")
    print("\n[ALERT] Data Drift Detected!")
    print(f"Drift Score: {drift_score:.2f}")
    print(f"Drifting Features: {drifted_features}")


def run_monitoring():
    # --- Load Data ---
    if not os.path.exists(PRODUCTION_FILE):
        logger.info("No production logs found — skipping monitoring.")
        print("No production logs found yet.")
        return

    logger.info("Loading reference and production data...")
    ref_data = pd.read_csv(REFERENCE_FILE)
    prod_data = pd.read_csv(PRODUCTION_FILE)

    features = [col for col in ref_data.columns if col != 'Target']
    current_data = prod_data[features]

    logger.info(f"Analyzing {len(current_data)} samples for drift.")

    # --- Create Initial Evidently Report ---
    report = Report(metrics=[DataDriftPreset()])
    eval = report.run(reference_data=ref_data, current_data=current_data)

    eval.save_html(REPORT_OUTPUT)
    eval.save_json(REPORT_JSON)
    logger.info(f"Saved main drift report: {REPORT_OUTPUT}")

    # ===================================================================
    #           FEATURE DRIFT EVALUATION
    # ===================================================================
    feature_cols = [
        'Volatility_Target', 'RSI_Target', 'SMA_Target', 'Return_Target', 'Volume_Target',
        'Volatility_SPY', 'RSI_SPY', 'SMA_SPY', 'Return_SPY', 'Volume_SPY'
    ]

    prediction_cols = ['Prediction', 'Target']

    reference_features = ref_data[feature_cols]
    current_features = prod_data[feature_cols]

    feature_report = Report(metrics=[DataDriftPreset()])
    data_eval = feature_report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    data_eval.save_html("reports/tsla_featuredrift_report.html")
    logger.info("Saved reports/tsla_featuredrift_report.html")

    # ===================================================================
    #           TARGET / PREDICTION DRIFT EVALUATION
    # ===================================================================
    reference_prediction = ref_data[prediction_cols]
    current_prediction = prod_data[prediction_cols]

    pred_eval = feature_report.run(
        reference_data=reference_prediction,
        current_data=current_prediction
    )

    pred_eval.save_html("reports/tsla_targetdrift_report.html")
    logger.info("Saved reports/tsla_targetdrift_report.html")

    # ===================================================================
    #           PROGRAMMATIC DRIFT CHECKING
    # ===================================================================
    data_eval.save_json("reports/tsla_datadrift_report.json")
    results = data_eval.dict()

    drift_metric = results["metrics"][0]  # DriftedColumnsCount metric
    drift_share = drift_metric["value"]["share"]
    num_drifted = drift_metric["value"]["count"]

    logger.info(f"Drift Share: {drift_share}, Drifted Features: {num_drifted}")
    print(f"Drift Share: {drift_share}, Drifted Features: {num_drifted}")

    # Per-feature drift check
    drifted_columns = []

    for metric in results["metrics"]:
        if metric["config"]["type"] == "evidently:metric_v2:ValueDrift":
            col = metric["config"]["column"]
            score = metric["value"]
            threshold = metric["config"]["threshold"]

            logger.debug(f"Checking drift: {col} (score={score}, threshold={threshold})")

            if score > threshold:
                drifted_columns.append(col)

    # --- Decision Logic ---
    if drift_share > DRIFT_THRESHOLD:
        logger.warning(f"Significant drift detected. Columns: {drifted_columns}")
        print("[ALERT] Significant drift detected!")
        print("Drifted columns:", drifted_columns)

        send_alert(drift_share, drifted_columns)
        trigger_retraining()
    else:
        logger.info("Healthy — no significant drift detected.")
        print("[STATUS] Healthy — no significant drift detected.")


if __name__ == "__main__":
    logger.info("=== Drift Monitoring Started ===")
    run_monitoring()
    logger.info("=== Drift Monitoring Completed ===")
