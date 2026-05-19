import pandas as pd
import numpy as np
import xgboost as xgb
import re
from google.cloud import bigquery
from google.cloud import storage

def run_daily_inference():
    # Set up GCP clients to talk to BigQuery and Cloud Storage
    bq_client = bigquery.Client(project="vehicle-fleet-prognostics")
    storage_client = storage.Client(project="vehicle-fleet-prognostics")

    # Pull the daily model from the bronze bucket
    bucket = storage_client.bucket("scania-telemetry-raw-bronze-bucket-ba")
    blob = bucket.blob("models/sota_scania_xgboost_v1.json")
    blob.download_to_filename("/tmp/model.json")
    
    bst = xgb.Booster()
    bst.load_model("/tmp/model.json")

    # Grabbing today's batch from the silver layer
    query = "SELECT * FROM `vehicle-fleet-prognostics.fleet_telemetry_silver.aps_failures_raw` LIMIT 1000"
    df_daily = bq_client.query(query).to_dataframe()
    df_daily['Truck_ID'] = df_daily.index 

    # Meta-feature engineering 
    # The MNAR (Missing Not At Random) signal is massive here, so we capture the count of NaNs and zeros.
    features = df_daily.drop(columns=['class', 'Truck_ID'], errors='ignore').apply(pd.to_numeric, errors='coerce')
    features['META_missing_count'] = features.isnull().sum(axis=1)
    features['META_zero_count'] = (features == 0).sum(axis=1)
    features['META_sensor_std'] = features.std(axis=1)
    
    # Dropping columns with near 100% missingness to reduce noise
    cols_to_drop = ['ab_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000', 'br_000', 'cr_000']
    features_cleaned = features.drop(columns=cols_to_drop, errors='ignore')
    
    # Build binary indicators for missing sensors
    missing_indicators = {f"{col}_is_missing": features_cleaned[col].isnull().astype(int) 
                          for col in features_cleaned.columns if features_cleaned[col].isnull().any() and not col.startswith('META')}
    
    X = pd.concat([features_cleaned, pd.DataFrame(missing_indicators)], axis=1) if missing_indicators else features_cleaned
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))

    # XGBoost throws a shape mismatch error if the daily batch is missing columns that existed in training.
    # Force-fill any missing feature columns with 0 to align the matrices.
    expected_features = bst.feature_names
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0 
            
    X = X[expected_features]

    # Predict failure probability
    dmatrix = xgb.DMatrix(X)
    preds_logits = bst.predict(dmatrix)
    preds_risk = 1.0 / (1.0 + np.exp(-preds_logits))

    # Apply the $9,490 optimized threshold derived from the cost-matrix analysis
    df_daily['Failure_Risk'] = preds_risk
    df_daily['Needs_Inspection'] = (preds_risk >= 0.0102).astype(int)
    
    df_gold = df_daily[df_daily['Needs_Inspection'] == 1][['Truck_ID', 'Failure_Risk']]

    # Dump the flagged trucks into the gold table for the morning maintenance report
    gold_table_id = "vehicle-fleet-prognostics.fleet_telemetry_gold.daily_high_risk_trucks"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    bq_client.load_table_from_dataframe(df_gold, gold_table_id, job_config=job_config).result()
    
    return f"Successfully flagged {len(df_gold)} trucks for maintenance."

if __name__ == "__main__":
    run_daily_inference()