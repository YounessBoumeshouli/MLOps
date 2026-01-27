"""
Unit tests for the FastAPI application.

Tests cover:
- Health check endpoint
- Prediction endpoint
- Metrics endpoint
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.api.main import app, model_cache


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Reset model cache before each test."""
    model_cache["model"] = None
    model_cache["version"] = None
    model_cache["loaded_at"] = None
    yield
    model_cache["model"] = None
    model_cache["version"] = None
    model_cache["loaded_at"] = None


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check when everything is working."""
        with patch('src.api.main.check_mlflow_connection', return_value=True):
            with patch.object(model_cache, '__getitem__', side_effect=lambda x: "1" if x == "version" else Mock()):
                response = client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert "mlflow_connected" in data
                assert "model_loaded" in data
    
    def test_health_check_mlflow_disconnected(self, client):
        """Test health check when MLflow is not connected."""
        with patch('src.api.main.check_mlflow_connection', return_value=False):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["mlflow_connected"] is False
            assert data["status"] == "degraded"


class TestPredictionEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_success(self, client, mock_model):
        """Test successful prediction."""
        model_cache["model"] = mock_model
        model_cache["version"] = "1"
        
        request_data = {
            "features": [
                1.5, -0.3, 2.1, 0.8, -1.2,
                0.5, 1.8, -0.7, 2.3, 0.2,
                -1.5, 1.1, 0.9, -0.4, 2.0,
                0.7, -1.8, 1.3, 0.6, -0.9
            ]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert data["prediction"] == 1
        assert len(data["probability"]) == 2
    
    def test_predict_invalid_features_count(self, client):
        """Test prediction with wrong number of features."""
        request_data = {
            "features": [1.0, 2.0, 3.0]  # Only 3 features instead of 20
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_empty_features(self, client):
        """Test prediction with empty features list."""
        request_data = {
            "features": []
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_model_not_loaded(self, client):
        """Test prediction when model is not loaded."""
        with patch('src.api.main.load_model_from_mlflow', return_value=(None, None)):
            request_data = {
                "features": [
                    1.5, -0.3, 2.1, 0.8, -1.2,
                    0.5, 1.8, -0.7, 2.3, 0.2,
                    -1.5, 1.1, 0.9, -0.4, 2.0,
                    0.7, -1.8, 1.3, 0.6, -0.9
                ]
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 503  # Service unavailable
    
    def test_predict_loads_model_on_demand(self, client, mock_model):
        """Test that model is loaded on demand if not in cache."""
        with patch('src.api.main.load_model_from_mlflow', return_value=(mock_model, "1")):
            request_data = {
                "features": [
                    1.5, -0.3, 2.1, 0.8, -1.2,
                    0.5, 1.8, -0.7, 2.3, 0.2,
                    -1.5, 1.1, 0.9, -0.4, 2.0,
                    0.7, -1.8, 1.3, 0.6, -0.9
                ]
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
            assert model_cache["model"] is not None


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test that metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for some expected metrics
        content = response.text
        assert "api_requests_total" in content or "python_info" in content


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""
    
    def test_model_info_not_loaded(self, client):
        """Test model info when no model is loaded."""
        response = client.get("/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is False
    
    def test_model_info_loaded(self, client, mock_model):
        """Test model info when model is loaded."""
        model_cache["model"] = mock_model
        model_cache["version"] = "1"
        model_cache["loaded_at"] = 1234567890.0
        
        response = client.get("/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is True
        assert data["version"] == "1"
        assert "loaded_at" in data
