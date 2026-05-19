# Predictive Maintenance in Heavy-Duty Commercial Fleets: A Cost-Optimized Machine Learning Approach

## Overview
This repository contains the source code and cloud deployment architecture for an automated predictive maintenance pipeline. The primary objective is to identify impending Air Pressure System (APS) failures in heavy Scania trucks using high-dimensional, highly sparse telemetry data. By shifting from a reactive maintenance schedule to a proactive, model-driven approach, this project demonstrates a quantifiable reduction in operational fleet costs.

## Methodology
The sensor dataset presents a classic challenge in industrial telemetry: extreme data sparsity. Rather than treating missing values purely as noise to be imputed, the absence of data was treated as a Missing-Not-At-Random (MNAR) signal, which is often indicative of physical sensor degradation or localized mechanical failure. To capture this, the feature engineering pipeline constructs meta-features that aggregate row-wise missingness and zero-value frequencies, alongside binary indicator variables for sensors exhibiting partial observability. An XGBoost classifier was selected for the primary modeling strategy due to its native sparsity-aware split finding and computational efficiency on complex tabular data. 

## Cost-Matrix Optimization
Traditional classification metrics, such as the F1-score or standard area under the curve, were deemed insufficient for this specific deployment due to the highly asymmetric financial impact of classification errors. Instead, the predictive decision threshold was explicitly calibrated against a custom business cost matrix. This calculated threshold of 0.0102 mathematically balances the relatively low cost of a preventative mechanic inspection against the severe financial penalty of a catastrophic, on-road vehicle failure.

## Cloud Infrastructure and Reproducibility
The data pipeline is built on a serverless Google Cloud Platform Medallion architecture to ensure reproducibility, scalability, and automated batch processing. Raw telemetry logs and serialized model weights are initially staged in a Google Cloud Storage bronze layer before being queried from a BigQuery silver dataset. For the daily batch inference, a containerized Python application pulls the latest logs for 1,000 trucks, applies the necessary feature transformations, and executes inference to calculate a failure probability for each vehicle. Trucks exceeding the optimized risk threshold are then written to a finalized BigQuery gold table utilized for generating daily maintenance manifests. 

## Automated Orchestration
This entire infrastructure is automated and orchestrated via Google Cloud Run Jobs. A Cloud Scheduler trigger initiates the pipeline nightly at 02:00 UTC. The containerized service executes the full data extraction, transformation, and load process within a strict 2GB memory constraint, and subsequently scales back to zero immediately post-execution to drastically minimize persistent compute overhead.

## Empirical Results
Evaluation on the holdout dataset demonstrates that the cost-optimized pipeline reduced simulated maintenance costs from a baseline of $35,000 down to $9,490. This represents an estimated 72.8% reduction in failure-related expenditures, proving the viability of deploying highly specialized, cost-aware machine learning models in heavy industrial environments.
