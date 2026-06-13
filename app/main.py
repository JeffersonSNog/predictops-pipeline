from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel
import pandas as pd
from src.preprocess import encode_type, engineer_features

app = FastAPI(title="PredictOps API", version="1.0.0")

class MachineFeatures(BaseModel):
    Type: str
    Air_Temperature: float
    Process_Temperature: float
    Rotational_Speed: int
    Torque: float
    Tool_Wear: float

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
    return {"prediction": int(prediction), "probability": float(probability)}

