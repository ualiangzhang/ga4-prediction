# GA4-Based Conversion Prediction and ML Pipeline

This project demonstrates a complete machine learning workflow for **user conversion prediction** based on **Google Analytics 4 (GA4) event data**. It showcases the end-to-end process from data extraction and feature engineering to model training, interpretation, and cloud deployment.

---

## 🔍 Project Overview

- 🎯 **Goal**: Predict whether a user will convert (i.e., make a purchase) based on historical behavioral data.
- 📊 **Data**: GA4 sample e-commerce event dataset from BigQuery public datasets.
- 🛠️ **Tech Stack**:
  - **BigQuery SQL** for feature table generation
  - **Pandas + XGBoost** for model development
  - **Prophet / ARIMA / LSTM** for time series forecasting
  - **Transformers + Scikit-learn** for NLP applications
  - **Vertex AI / Azure ML** for deployment and MLOps

---

## 🗂️ Project Structure

```bash
ga4-prediction-project/
├── etl/                 # SQL queries and BigQuery export scripts
├── data/                # Raw + processed datasets (excluded via .gitignore)
├── src/                 # Core Python modules (feature, model, nlp, ts)
├── notebooks/           # Jupyter demos for exploration and reporting
├── deployment/          # Deployment scripts for GCP / Azure
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
