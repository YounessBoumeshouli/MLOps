# train.py - Your script is already good, just minor tweaks:
import pandas as pd

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():
    # MLflow configuration
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    experiment_name = "ml_model_training"
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    logger.info(f"Experiment: {experiment_name}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_v1") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Generate sample data (simulating your dataset)
        df = pd.read_csv("data/data.csv")
        X = df.drop(columns=['cluster'])
        y = df['cluster']
        X_train, X_test,    y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=104, shuffle=True)

        # Define model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        }
        
       
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")
        
  
        logger.info("Training model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
     
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        
    
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")
        
        # Save model locally
        os.makedirs('models', exist_ok=True)
        model_path = 'models/model.joblib'
        joblib.dump(model, model_path)
        
        # ✅ TASK 3: Log model artifact and register
        model_name = "ml_classifier"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        logger.info(f"✓ Model logged to MLflow and registered as '{model_name}'")
        
        # ✅ TASK 4: Get model version
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if model_versions:
            latest_version = model_versions[0].version
            logger.info(f"Model version: {latest_version}")
            
            # ✅ TASK 5: Promote to Staging
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Staging"
            )
            logger.info(f"✓ Model v{latest_version} promoted to STAGING")
            
            # ✅ TASK 6: Promote to Production
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f"✓ Model v{latest_version} promoted to PRODUCTION")
            
            return run.info.run_id, latest_version
        
        return run.info.run_id, None


if __name__ == "__main__":
    run_id, version = train_model()
    print(f"\n{'='*60}")
    print(f"✓ Training completed!")
    print(f"  Run ID: {run_id}")
    print(f"  Model Version: {version}")
    print(f"  Status: PRODUCTION")
    print(f"{'='*60}\n")