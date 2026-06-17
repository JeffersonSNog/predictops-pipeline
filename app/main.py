from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel, Field
import pandas as pd
from src.preprocess import encode_type, engineer_features
from datetime import datetime
from typing import Literal

app = FastAPI(title="PredictOps API", version="1.0.0")

model = None

class MachineFeatures(BaseModel):
    Type: Literal["L", "M", "H"]
    Air_Temperature: float = Field(ge=0)
    Process_Temperature: float = Field(ge=0)
    Rotational_Speed: int = Field(ge=0)
    Torque: float = Field(ge=0)
    Tool_Wear: float = Field(ge=0)

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model("runs:/bd8f8ff503ab4824b2dac683d5920a7f/model")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: MachineFeatures):
    df = pd.DataFrame([features.dict()])
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
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "prediction": int(prediction),
        "label": "FAILURE" if prediction == 1 else "NORMAL",
        "probability": float(probability)
    }

@app.get("/model-info")
def modelinfo():
    return({"Nome do Modelo" : "Gradient Boosting",
           "F1": 0.89,
           "Roc_Auc": 0.97,
           "Data": datetime.now().isoformat()})
