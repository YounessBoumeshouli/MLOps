"""
Unit tests for the training script.

Tests cover:
- MLflow integration
- Model training
- Model logging
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import mlflow


class TestTraining:

    @patch('src.train.mlflow')
    @patch('src.train.RandomForestClassifier')
    def test_train_model_logs_to_mlflow(self, mock_classifier, mock_mlflow):
        """Test that training logs parameters and metrics to MLflow."""
        from src.train import train_model
        
        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.search_model_versions.return_value = [mock_version]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        # Mock classifier
        mock_model = MagicMock()
        mock_classifier.return_value = mock_model
        
        # Run training
        with patch.dict('os.environ', {'MLFLOW_TRACKING_URI': 'http://localhost:5000'}):
            run_id, version = train_model()
        
        # Verify MLflow calls
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.set_experiment.called
        assert mock_mlflow.log_params.called
        assert mock_mlflow.log_metrics.called
        
        # Verify model was trained
        assert mock_model.fit.called
    
    @patch('src.train.mlflow')
    def test_train_model_registers_model(self, mock_mlflow):
        """Test that model is registered in MLflow."""
        from src.train import train_model
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.search_model_versions.return_value = [mock_version]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        with patch.dict('os.environ', {'MLFLOW_TRACKING_URI': 'http://localhost:5000'}):
            run_id, version = train_model()
        
        # Verify model was logged
        assert mock_mlflow.sklearn.log_model.called
        
        # Verify model was transitioned to Production
        assert mock_client.transition_model_version_stage.called
