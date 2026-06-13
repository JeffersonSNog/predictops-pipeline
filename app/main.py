from fastapi import FastAPI
import mlflow.sklearn

app = FastAPI(title="PredictOps API", version="1.0.0")

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model("runs:/bd8f8ff503ab4824b2dac683d5920a7f/model")


@app.get("/health")
def health():
    return {"status": "ok"}