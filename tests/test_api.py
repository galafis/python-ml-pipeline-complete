from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")  # Rota corrigida para o health check
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    # Payload corrigido para corresponder ao schema PredictionRequest
    # O modelo LogisticRegression espera um array 2D de features numéricas.
    # O número de features deve ser consistente com o modelo treinado.
    # Para o dummy_data.csv, temos 4 features.
    payload = {"data": [[1.0, 2.0, 3.0, 4.0]]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

