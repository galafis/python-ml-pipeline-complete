from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    # Substitua pelo schema correto esperado pela sua API
    payload = {"feature1": 1, "feature2": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
