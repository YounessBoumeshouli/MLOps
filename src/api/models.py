"""
Pydantic models for API request/response validation.

These models provide:
- Type validation
- Automatic documentation in Swagger UI
- Request/response serialization
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    
    Attributes:
        features: List of numerical features for prediction
    """
    features: List[float] = Field(
        ...,
        description="List of numerical features for model prediction",
        example=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that features list is not empty and has correct length."""
        if len(v) == 0:
            raise ValueError("Features list cannot be empty")
        if len(v) != 20:  # Adjust based on your model's expected input
            raise ValueError(f"Expected 20 features, got {len(v)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    1.5, -0.3, 2.1, 0.8, -1.2,
                    0.5, 1.8, -0.7, 2.3, 0.2,
                    -1.5, 1.1, 0.9, -0.4, 2.0,
                    0.7, -1.8, 1.3, 0.6, -0.9
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint.
    
    Attributes:
        prediction: The predicted class (0 or 1)
        probability: Probability scores for each class
        model_version: Version of the model used for prediction
        timestamp: When the prediction was made
    """
    prediction: int = Field(
        ...,
        description="Predicted class label",
        example=1
    )
    probability: List[float] = Field(
        ...,
        description="Probability scores for each class",
        example=[0.3, 0.7]
    )
    model_version: str = Field(
        ...,
        description="Version of the model used",
        example="1"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the prediction"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": [0.25, 0.75],
                "model_version": "1",
                "timestamp": "2024-01-27T12:00:00"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Overall health status
        mlflow_connected: Whether MLflow is accessible
        model_loaded: Whether the production model is loaded
        timestamp: Current server time
    """
    status: str = Field(
        ...,
        description="Overall health status",
        example="healthy"
    )
    mlflow_connected: bool = Field(
        ...,
        description="MLflow connectivity status",
        example=True
    )
    model_loaded: bool = Field(
        ...,
        description="Whether production model is loaded",
        example=True
    )
    model_version: Optional[str] = Field(
        None,
        description="Loaded model version",
        example="1"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "mlflow_connected": True,
                "model_loaded": True,
                "model_version": "1",
                "timestamp": "2024-01-27T12:00:00"
            }
        }


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    
    Attributes:
        error: Error message
        detail: Detailed error information
        timestamp: When the error occurred
    """
    error: str = Field(
        ...,
        description="Error message",
        example="Prediction failed"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information",
        example="Model not found in MLflow"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
