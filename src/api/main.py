"""
FastAPI application for ML model serving with MLflow integration.

This API provides:
- /predict: Model inference endpoint
- /health: Health check with MLflow connectivity
- /metrics: Prometheus metrics exposure
- Automatic Swagger documentation at /docs
"""

import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .models import PredictionRequest, PredictionResponse, HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
MODEL_ACCURACY = Gauge(
    'model_training_accuracy',
    'Model training accuracy score',
    ['model_version']
)

MODEL_F1_SCORE = Gauge(
    'model_training_f1_score',
    'Model training F1 score',
    ['model_version']
)

MODEL_PRECISION = Gauge(
    'model_training_precision',
    'Model training precision score',
    ['model_version']
)

MODEL_RECALL = Gauge(
    'model_training_recall',
    'Model training recall score',
    ['model_version']
)
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_version']
)

PREDICTION_DURATION = Histogram(
    'prediction_duration_seconds',
    'Prediction duration in seconds'
)

ERROR_COUNT = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['endpoint', 'error_type']
)

# Global model cache
model_cache = {
    "model": None,
    "version": None,
    "loaded_at": None
}


def load_model_from_mlflow(model_name: str = "ml_classifier", stage: str = "Production"):
    """
    Load a model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, None)
    
    Returns:
        Loaded model and version information
    """
    try:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        logger.info(f"Loading model '{model_name}' from stage '{stage}'")
        
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get model version info
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        
        if model_versions:
            version = model_versions[0].version
            run_id = model_versions[0].run_id
            logger.info(f"Successfully loaded model version {version} (Run ID: {run_id})")
            
            # Fetch run metrics
            try:
                run = client.get_run(run_id)
                metrics = run.data.metrics
                
                # Update Prometheus gauges
                if "accuracy" in metrics:
                    MODEL_ACCURACY.labels(model_version=version).set(metrics["accuracy"])
                if "f1_score" in metrics:
                    MODEL_F1_SCORE.labels(model_version=version).set(metrics["f1_score"])
                if "precision" in metrics:
                    MODEL_PRECISION.labels(model_version=version).set(metrics["precision"])
                if "recall" in metrics:
                    MODEL_RECALL.labels(model_version=version).set(metrics["recall"])
                    
                logger.info(f"Updated Prometheus metrics for version {version}")
            except Exception as e:
                logger.warning(f"Failed to fetch metrics for run {run_id}: {e}")

            return model, version
        else:
            logger.warning(f"No model found in stage '{stage}'")
            return None, None
            
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        return None, None


def check_mlflow_connection() -> bool:
    """Check if MLflow tracking server is accessible."""
    try:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Try to list experiments
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        return True
    except Exception as e:
        logger.error(f"MLflow connection check failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads the model on startup.
    """
    # Startup: Load model
    logger.info("Starting up API server...")
    
    # Wait a bit for MLflow to be ready (in Docker environment)
    time.sleep(2)
    
    model, version = load_model_from_mlflow()
    if model:
        model_cache["model"] = model
        model_cache["version"] = version
        model_cache["loaded_at"] = time.time()
        logger.info(f"Model loaded successfully: version {version}")
    else:
        logger.warning("Failed to load model on startup. Will retry on first prediction.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="MLOps API",
    description="Machine Learning model serving API with MLflow integration",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to track request metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MLOps API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with system status
    """
    mlflow_connected = check_mlflow_connection()
    model_loaded = model_cache["model"] is not None
    
    status = "healthy" if (mlflow_connected and model_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        mlflow_connected=mlflow_connected,
        model_loaded=model_loaded,
        model_version=model_cache["version"]
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction using the loaded model.
    
    Args:
        request: PredictionRequest with features
    
    Returns:
        PredictionResponse with prediction results
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model_cache["model"] is None:
            logger.info("Model not in cache, attempting to load...")
            model, version = load_model_from_mlflow()
            
            if model is None:
                ERROR_COUNT.labels(endpoint="/predict", error_type="model_not_found").inc()
                raise HTTPException(
                    status_code=503,
                    detail="Model not available. Please ensure a model is registered in MLflow Production stage."
                )
            
            model_cache["model"] = model
            model_cache["version"] = version
            model_cache["loaded_at"] = time.time()
        
        # Prepare input data
        input_data = np.array([request.features])
        
        # Make prediction
        prediction = model_cache["model"].predict(input_data)
        
        # Get prediction probabilities if available
        try:
            probabilities = model_cache["model"].predict_proba(input_data)[0].tolist()
        except AttributeError:
            # Model doesn't support predict_proba
            probabilities = [1.0 if i == prediction[0] else 0.0 for i in range(2)]
        
        # Record metrics
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)
        PREDICTION_COUNT.labels(model_version=model_cache["version"]).inc()
        
        logger.info(f"Prediction made: {prediction[0]} (took {duration:.3f}s)")
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=probabilities,
            model_version=str(model_cache["version"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict", error_type="prediction_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics in text format
    """
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model-info", tags=["Model"])
async def model_info():
    """
    Get information about the currently loaded model.
    
    Returns:
        Model version and loading information
    """
    if model_cache["model"] is None:
        return {
            "loaded": False,
            "message": "No model currently loaded"
        }
    
    return {
        "loaded": True,
        "version": model_cache["version"],
        "loaded_at": model_cache["loaded_at"],
        "model_name": "ml_classifier"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
