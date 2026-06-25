from fastapi import FastAPI
from contextlib import asynccontextmanager
import mlflow.sklearn
from pydantic import BaseModel, Field
import pandas as pd
from src.preprocess import encode_type, engineer_features
from datetime import datetime
from typing import Literal
import joblib

model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    scaler = joblib.load("mlruns/185296854917139986/2e7d81a853b643de99cda002a82cb395/artifacts/scaler.pkl")
    model = mlflow.sklearn.load_model("mlruns/185296854917139986/2e7d81a853b643de99cda002a82cb395/artifacts/model")
    yield

app = FastAPI(title="PredictOps API", version="1.0.0", lifespan=lifespan)

class MachineFeatures(BaseModel):
    Type: Literal["L", "M", "H"]
    Air_Temperature: float = Field(ge=0)
    Process_Temperature: float = Field(ge=0)
    Rotational_Speed: int = Field(ge=0)
    Torque: float = Field(ge=0)
    Tool_Wear: float = Field(ge=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: MachineFeatures):
    df = pd.DataFrame([features.model_dump()])
    df = df.rename(columns={
    "Type": "Type",
    "Air_Temperature": "Air temperature [K]",
    "Process_Temperature": "Process temperature [K]",
    "Rotational_Speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool_Wear": "Tool wear [min]"
})
    df = encode_type(df)
    df = engineer_features(df)
    df = scaler.transform(df)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "prediction": int(prediction),
        "label": "FAILURE" if prediction == 1 else "NORMAL",
        "probability": float(probability)
    }

@app.get("/model-info")
def modelinfo():
    return({"model_name" : "Gradient Boosting",
           "f1": 0.88,
           "roc_auc": 0.96,
           "timestamp": datetime.now().isoformat()})
