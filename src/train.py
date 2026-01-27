"""
Training script with MLflow integration for experiment tracking and model registry.

This script demonstrates how to:
- Track experiments with MLflow
- Log parameters, metrics, and artifacts
- Register models in MLflow Model Registry
- Promote models through stages (Staging -> Production)
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():

    
    # MLflow configuration
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    experiment_name = "ml_model_training"
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    logger.info(f"Experiment: {experiment_name}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="training_run") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        logger.info("Parameters logged to MLflow")
        
        # Generate synthetic dataset (replace with your actual data loading)
        logger.info("Generating dataset...")
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X.shape[1])
        
        # Train model
        logger.info("Training model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")
        
        # Log model
        model_name = "ml_classifier"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        logger.info(f"Model logged to MLflow with name: {model_name}")
        
        # Get the model version
        client = mlflow.tracking.MlflowClient()
        model_version = client.search_model_versions(f"name='{model_name}'")[0].version
        
        logger.info(f"Model version: {model_version}")
        
        # Promote model to Production
        try:
            # Transition to Staging first
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging"
            )
            logger.info(f"Model version {model_version} transitioned to Staging")
            
            # Then to Production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f"Model version {model_version} transitioned to Production")
            
        except Exception as e:
            logger.error(f"Error transitioning model: {e}")
        
        logger.info("Training completed successfully!")
        
        return run.info.run_id, model_version


if __name__ == "__main__":
    run_id, version = train_model()
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Run ID: {run_id}")
    print(f"Model Version: {version}")
    print(f"{'='*60}\n")
