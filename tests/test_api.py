from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_checck():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_positive():
    payload = {"tweet": "frank edwards eh! he sabi sing"}
    response = client.post("/predict", json=payload)
    assert response.status - code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert data["label"] in ["Positive", "Neutral", "Negative"]
