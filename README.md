# GA4 Prediction Pipeline

An end-to-end ML pipeline for **Google Analytics 4 (GA4)** data: ETL, classification & forecasting, hyperparameter tuning, REST API service, and Docker deployment.

---

## 🚀 Features

| Component                | Description                                                                            | Tech Stack                                 |
|--------------------------|----------------------------------------------------------------------------------------|--------------------------------------------|
| **ETL**                  | - Export data from GA4<br>- Build user-level & hourly feature tables                   | BigQuery · Python (pandas)                 |
| **Classification**       | - Predict user conversion (purchase vs. no-purchase)<br>- Models: XGBoost, RF, LR, DNN | scikit-learn · XGBoost · Keras/TensorFlow  |
| **Forecasting**          | - Hourly traffic forecasting<br>- Models: ARIMA, Prophet, LSTM                          | statsmodels · Prophet · Keras/TensorFlow   |
| **Hyperparameter Tuning**| - Optimize model parameters with Optuna                                                | Optuna                                     |
| **API Service**          | - FastAPI endpoints `/classify` & `/forecast`                                          | FastAPI · Uvicorn                          |
| **Deployment**           | - Dockerized API for local or cloud                                                    | Docker                                     |
| **Testing**              | - Unit tests for ETL, training, inference, API                                         | pytest                                     |

---

## 📦 Installation

```bash
git clone https://github.com/ualiangzhang/ga4-prediction.git
cd ga4-prediction
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧩 Project Structure

```
ga4-prediction/
├── api/                         # FastAPI app (main.py)
├── etl/
│   ├── scripts/                 # export_ga4.py, preprocess.py
│   └── data/
│       ├── raw/                 # GA4 CSV exports
│       └── processed/           # Feature tables
├── models/                      # Saved model artifacts & params
├── src/
│   ├── classification/          # train.py, predict.py, utils
│   └── timeseries/              # train.py, predict.py, utils
├── tests/                       # pytest unit tests
├── tune/                        # Optuna tuning scripts
├── Dockerfile                   # Container definition
└── requirements.txt             # Python dependencies
```

---

## 🔧 Usage

### 1. ETL

```bash
python etl/scripts/export_ga4.py      # Export raw data → etl/data/raw/
python etl/scripts/preprocess.py      # Generate features → etl/data/processed/
```

### 2. Classification

```bash
# Train XGBoost classifier
python src/classification/train.py --model xgb

# Batch inference
python src/classification/predict.py   --input etl/data/processed/ga4_training_data.csv   --model-path models/xgb_model.pkl
```

### 3. Forecasting

```bash
# Train Prophet model
python src/timeseries/train.py --model prophet

# Forecast next 24h
python src/timeseries/predict.py   --input etl/data/processed/hourly_history.csv   --model-path models/prophet_model.pkl
```

### 4. Hyperparameter Tuning

```bash
python tune/tune.py --model xgb --trials 50
```

---

## 🧪 Testing

```bash
pytest
```

---

## 📡 API Endpoints

Start server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

- **POST** `/classify`  
  - **Request**: JSON with feature records & model path  
  - **Response**: Predicted labels & probabilities

- **POST** `/forecast`  
  - **Request**: JSON with hourly history & model path  
  - **Response**: Forecasted timestamps & values

Swagger UI: `http://localhost:8080/docs`

---

## 🐳 Docker Deployment

**Build image**

```bash
docker build -t ga4-predict:latest .
```

**Run container**

```bash
docker run --rm -it   -p 8080:8080   --ulimit nproc=2048:2048   ga4-predict:latest
```

Add these lines to `Dockerfile` if not already present to prevent OpenBLAS crashes:

```dockerfile
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV MALLOC_CONF=background_thread:false
```

---

## 🚀 Deploy to Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/<PROJECT_ID>/ga4-predict
gcloud run deploy ga4-predict   --image gcr.io/<PROJECT_ID>/ga4-predict   --region us-central1   --platform managed   --allow-unauthenticated
```

---

## 📜 License

Apache 2.0
