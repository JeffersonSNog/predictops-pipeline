# predictops-pipeline

MLOps pipeline for predictive maintenance on industrial machines — detecting failures before they happen.

Built with FastAPI, MLflow, Docker, and GitHub Actions CI/CD.

---

## Architecture

```
Raw Data (AI4I 2020)
       │
       ▼
Preprocessing & Feature Engineering
       │
       ▼
Model Training (MLflow Tracking)
       │
       ▼
Champion Model: Gradient Boosting
       │
       ▼
FastAPI REST API  ◄──  StandardScaler (Train-Serving Skew prevention)
       │
       ▼
Docker Container
       │
       ▼
GitHub Actions CI/CD  →  Docker Hub
```

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Serialization | joblib |
| Testing | pytest |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Registry | Docker Hub |

---

## How to Run

### Local (with Docker Compose)

```bash
git clone https://github.com/JeffersonSNog/predictops-pipeline.git
cd predictops-pipeline
docker compose up --build
```

API available at: `http://localhost:8000`

### Local (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Pull from Docker Hub

```bash
docker pull jeffersonsnog/predictops-api:latest
docker run -p 8000:8000 jeffersonsnog/predictops-api:latest
```

---

## Endpoints

### `GET /health`
Returns API status.

```json
{ "status": "ok" }
```

### `POST /predict`
Predicts machine failure based on sensor readings.

**Request body:**
```json
{
  "Type": "L",
  "Air_Temperature": 298.0,
  "Process_Temperature": 308.0,
  "Rotational_Speed": 1500,
  "Torque": 35.0,
  "Tool_Wear": 5.0
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "NORMAL",
  "probability": 0.0011
}
```

### `GET /model-info`
Returns champion model metadata.

```json
{
  "model_name": "Gradient Boosting",
  "f1": 0.88,
  "roc_auc": 0.96,
  "timestamp": "2026-06-25T00:00:00"
}
```

---

## Model Metrics

| Model | F1 Score | ROC-AUC |
|---|---|---|
| Gradient Boosting ✅ | 0.88 | 0.96 |

Dataset: [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository

---

## CI/CD

| Workflow | Trigger | Action |
|---|---|---|
| CI | push / pull_request | Run unit tests with pytest |
| CD | push to main | Build and push Docker image to Docker Hub |

[![CI](https://github.com/JeffersonSNog/predictops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/JeffersonSNog/predictops-pipeline/actions/workflows/ci.yml)
[![CD](https://github.com/JeffersonSNog/predictops-pipeline/actions/workflows/cd.yml/badge.svg)](https://github.com/JeffersonSNog/predictops-pipeline/actions/workflows/cd.yml)

---

## Project Structure

```
predictops-pipeline/
├── .github/workflows/   # CI/CD pipelines
├── app/
│   └── main.py          # FastAPI application
├── data/
│   └── raw/             # AI4I 2020 dataset
├── mlruns/              # MLflow experiment tracking
├── notebooks/           # Exploratory data analysis
├── src/
│   ├── train.py         # Model training
│   ├── evaluate.py      # Model evaluation
│   └── preprocess.py    # Feature engineering
├── tests/               # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Author

**Jefferson S. Nog**  
[GitHub](https://github.com/JeffersonSNog) • [Portfolio](https://jsn.dev.br)