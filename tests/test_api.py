import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    result = client.get("/health")
    assert result.status_code == 200

def test_predict(client):
    result = client.post("/predict", json={"Type": "L","Air_Temperature": 1,"Process_Temperature": 1,"Rotational_Speed": 1,"Torque": 1,"Tool_Wear": 1})
    data = result.json()
    assert result.status_code == 200
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data

def test_model_info(client):
    result = client.get("/model-info")
    data = result.json()
    assert result.status_code == 200
    assert "Nome do Modelo" in data
    assert "F1" in data
    assert "Roc_Auc" in data
    assert "Data" in data

def test_predict_invalid_type(client):
    result = client.post("/predict", json={"Type": "X","Air_Temperature": 1,"Process_Temperature": 1,"Rotational_Speed": 1,"Torque": 1,"Tool_Wear": 1})
    assert result.status_code == 422

def test_predict_invalid_number(client):
    result = client.post("/predict", json={"Type": "L","Air_Temperature": -1,"Process_Temperature": -1,"Rotational_Speed": -1,"Torque": -1,"Tool_Wear": -1})
    assert result.status_code == 422