# InsuranceClaim
This project is show how insurance claims are approved and investigated through MlOps

# Overview
Built an end-to-end MLOps pipeline for health insurance claim anomaly detection

The system identifies suspicious insurance claims and flags them for manual investigation, while normal claims can be approved automatically

Focused on model development, experiment tracking, model versioning, and production serving

# Step 1: Synthetic Data Generation

Generated synthetic health insurance claim data using Python (NumPy, Pandas)

Simulated realistic claim features:

Claim amount

Number of services

Patient age

Provider ID

Days since last claim

Introduced intentional anomalies (e.g., unusually high claim amounts and service counts) to represent potentially fraudulent claims

Shuffled and saved the dataset as a CSV file for reproducible training

Ensured deterministic results using a fixed random seed

# Step 2: Model Development & Training

Designed the problem as an unsupervised anomaly detection task

Selected Isolation Forest as the core model for detecting abnormal claim behavior

Performed train–test split to evaluate model behavior on unseen data

Engineered a clean feature set by excluding non-informative identifiers (e.g., claim ID)

# Step 3: Experiment Tracking with MLflow

Set up MLflow Tracking Server locally for experiment management

Created a dedicated experiment for insurance claim anomaly detection

Logged:

Model hyperparameters (n_estimators, contamination)

Training and test anomaly percentages

Trained model artifacts

Used MLflow to ensure reproducibility, traceability, and version control of models

Downloaded the trained model artifact for deployment

# Step 4: Model Packaging & Serving with BentoML

Packaged the trained anomaly detection model using BentoML

Registered the model for standardized, reusable serving

Enabled easy model versioning and deployment readiness

Abstracted model logic behind a service interface

# Step 5: API Layer for End Users

Built a Flask-based REST API for end-user interaction

The API:

Accepts claim details as input

Sends data to the BentoML-served model

Returns anomaly detection results (approve vs. investigate)

Designed the service to simulate real-world insurance claim submission workflows
