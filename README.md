# Heavy Fleet Predictive Maintenance Pipeline 🚛

An automated, serverless machine learning pipeline designed to predict catastrophic Air Pressure System (APS) failures in heavy-duty commercial fleets (Scania dataset). 

This project shifts fleet maintenance from reactive to proactive, utilizing a highly asymmetric cost-matrix to optimize the decision threshold. The final model **reduced simulated fleet maintenance costs by over 70%** compared to a baseline strategy.

## 🏗️ Cloud Architecture
This pipeline is deployed entirely on Google Cloud Platform (GCP) utilizing a Medallion data architecture:
* **Bronze Layer (GCS):** Raw model registry and raw, unstructured telemetry ingestion.
* **Silver Layer (BigQuery):** Cleaned, daily 170-sensor telemetry logs.
* **Gold Layer (BigQuery):** The final daily output table of high-risk Truck IDs prioritized for morning inspection.

## 🧠 Model & Inference
* **Algorithm:** XGBoost (C++ implementation for speed/sparsity). Consistently outperformed 2024 zero-shot transformers (TabPFN) and Kolmogorov-Arnold Networks (KAN) due to native handling of Missing-Not-At-Random (MNAR) tabular sensor data.
* **Feature Engineering:** Heavy emphasis on missingness indicators. In industrial sensors, a lack of data is often the strongest signal of mechanical failure.
* **Deployment:** Packaged in a Docker container and deployed via **Google Cloud Run Jobs**.
* **Orchestration:** Scheduled via **Cloud Scheduler** to wake up at 2:00 AM daily, run batch inference on the Silver BigQuery table, and write the flagged fleet IDs to the Gold table before shutting down.

## 📁 Repository Structure
* `main.py` - The core inference script that executes the daily cloud run.
* `Dockerfile` - Container instructions for the Cloud Run deployment.
* `requirements.txt` - Python dependencies.