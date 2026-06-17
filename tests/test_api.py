import pytest
from fastapi.testclient import TestClient
from app.main import app, load_model

load_model()
client = TestClient(app)

def test_health():
    result = client.get("/health")
    assert result.status_code == 200

def test_predict():
    result = client.post("/predict", json={"Type": "L","Air_Temperature": 1,"Process_Temperature": 1,"Rotational_Speed": 1,"Torque": 1,"Tool_Wear": 1})
    data = result.json()
    assert result.status_code == 200
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data

def test_model_info():
    result = client.get("/model-info")
    data = result.json()
    assert result.status_code == 200
    assert "Nome do Modelo" in data
    assert "F1" in data
    assert "Roc_Auc" in data
    assert "Data" in data

def test_predict_invalid_type():
    result = client.post("/predict", json={"Type": "X","Air_Temperature": 1,"Process_Temperature": 1,"Rotational_Speed": 1,"Torque": 1,"Tool_Wear": 1})
    assert result.status_code == 422

def test_predict_invalid_number():
    result = client.post("/predict", json={"Type": "L","Air_Temperature": -1,"Process_Temperature": -1,"Rotational_Speed": -1,"Torque": -1,"Tool_Wear": -1})
    assert result.status_code == 422