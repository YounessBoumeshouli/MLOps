# MLOps Complete Architecture

A production-ready MLOps platform for deploying Machine Learning models with comprehensive monitoring, experiment tracking, and CI/CD pipelines.

## 🏗️ Architecture

This project implements a complete MLOps stack with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐           │
│  │  FastAPI │───▶│  MLflow  │───▶│   Model    │           │
│  │   API    │    │ Registry │    │  Registry  │           │
│  └────┬─────┘    └──────────┘    └────────────┘           │
│       │                                                     │
│       ├──────────▶ ┌──────────────┐                       │
│       │            │  Prometheus  │                        │
│       │            └──────┬───────┘                        │
│       │                   │                                │
│       └──────────▶ ┌──────▼───────┐                       │
│                    │   Grafana    │                        │
│                    └──────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Technology Stack

- **API Framework**: FastAPI with Pydantic validation
- **ML Tracking**: MLflow (Experiment Tracking + Model Registry)
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Testing**: Pytest with coverage reporting
- **Code Quality**: Black, Flake8

## 📋 Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- Git
- (Optional) Python 3.11+ for local development

## 🔧 Installation & Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd MLOps
```

### 2. Start All Services

```bash
docker-compose up -d
```

This will start:
- **MLflow UI**: http://localhost:5000
- **FastAPI**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 3. Train and Register a Model

```bash
# Install dependencies locally (optional, for development)
pip install -r requirements.txt

# Run training script
python src/train.py
```

This will:
- Train a model
- Log parameters and metrics to MLflow
- Register the model in MLflow Model Registry
- Promote it to "Production" stage

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      1.5, -0.3, 2.1, 0.8, -1.2,
      0.5, 1.8, -0.7, 2.3, 0.2,
      -1.5, 1.1, 0.9, -0.4, 2.0,
      0.7, -1.8, 1.3, 0.6, -0.9
    ]
  }'

# View metrics
curl http://localhost:8000/metrics
```

### 5. Access Monitoring Dashboards

**Grafana Dashboard**:
1. Go to http://localhost:3000
2. Login with `admin` / `admin`
3. Navigate to "Dashboards" → "MLOps API Dashboard"
4. View real-time metrics: request rate, latency, errors, inference time

**MLflow UI**:
1. Go to http://localhost:5000
2. View experiments, runs, and model registry
3. Compare model versions and metrics

## 📁 Project Structure

```
MLOps/
├── src/
│   ├── train.py                 # Training script with MLflow integration
│   └── api/
│       ├── main.py              # FastAPI application
│       └── models.py            # Pydantic models
├── tests/
│   ├── test_api.py              # API tests
│   └── test_train.py            # Training tests
├── .github/workflows/
│   ├── ci.yml                   # Continuous Integration
│   └── cd.yml                   # Continuous Deployment
├── monitoring/
│   ├── prometheus.yml           # Prometheus configuration
│   └── grafana/
│       ├── dashboards/          # Pre-configured dashboards
│       └── provisioning/        # Auto-provisioning configs
├── Dockerfile                   # API container image
├── docker-compose.yml           # Multi-service orchestration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔄 MLflow Model Management

### Viewing Models

```bash
# Access MLflow UI
open http://localhost:5000
```

Navigate to "Models" to see registered models and their versions.

### Promoting Models

Models go through stages: `None` → `Staging` → `Production`

The training script automatically promotes models to Production. To manually promote:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

# Transition model to Production
client.transition_model_version_stage(
    name="ml_classifier",
    version="2",
    stage="Production",
    archive_existing_versions=True
)
```

### Loading Models in API

The API automatically loads the "Production" version of the model on startup and caches it for performance.

## 🔍 API Documentation

### Interactive API Docs

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check with MLflow status |
| `/predict` | POST | Make predictions |
| `/metrics` | GET | Prometheus metrics |
| `/model-info` | GET | Current model information |

### Example Requests

**Prediction**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.5, -0.3, 2.1, 0.8, -1.2, 0.5, 1.8, -0.7, 2.3, 0.2, -1.5, 1.1, 0.9, -0.4, 2.0, 0.7, -1.8, 1.3, 0.6, -0.9]
  }'
```

**Response**:
```json
{
  "prediction": 1,
  "probability": [0.25, 0.75],
  "model_version": "1",
  "timestamp": "2024-01-27T12:00:00"
}
```

## 📊 Monitoring & Metrics

### Prometheus Metrics

The API exposes the following metrics:

- `api_requests_total`: Total number of requests
- `api_request_duration_seconds`: Request latency histogram
- `predictions_total`: Total predictions made
- `prediction_duration_seconds`: Inference time histogram
- `api_errors_total`: Total errors by type

### Grafana Dashboard

The pre-configured dashboard includes:

- **Request Rate**: Requests per second by endpoint
- **Latency Percentiles**: P50, P95, P99 response times
- **Error Rate**: Percentage of failed requests
- **Inference Time**: Average model prediction time
- **Total Predictions**: Cumulative prediction count

## 🔄 CI/CD Pipeline

### Continuous Integration (CI)

Triggered on every push and pull request:

1. **Linting**: Black formatting check, Flake8 linting
2. **Testing**: Pytest with coverage reporting
3. **Docker Build**: Validates Docker image builds successfully

### Continuous Deployment (CD)

Triggered manually or on main branch merge:

1. **Build**: Creates optimized Docker image
2. **Tag**: Tags with version, branch, and SHA
3. **Push**: Pushes to GitHub Container Registry
4. **Deploy**: (Placeholder for your deployment strategy)

### GitHub Secrets Required

For CD pipeline to work, configure these secrets:

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

For custom deployment, you may need:
- `SERVER_HOST`: Deployment server address
- `SERVER_USER`: SSH username
- `SSH_PRIVATE_KEY`: SSH key for deployment

## 🧪 Development & Testing

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API locally
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/ --max-line-length=100
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run with markers
pytest tests/ -v -m "not slow"
```

## 🐛 Troubleshooting

### Issue: Model not found in MLflow

**Solution**: Train and register a model first:
```bash
python src/train.py
```

### Issue: API returns 503 Service Unavailable

**Cause**: No model in "Production" stage

**Solution**: 
1. Check MLflow UI at http://localhost:5000
2. Ensure a model is registered and in "Production" stage
3. Restart API: `docker-compose restart api`

### Issue: Grafana dashboard shows no data

**Solution**:
1. Verify Prometheus is scraping: http://localhost:9090/targets
2. Make some API requests to generate metrics
3. Check Prometheus datasource in Grafana settings

### Issue: Docker containers won't start

**Solution**:
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d

# Rebuild images
docker-compose build --no-cache
docker-compose up -d
```

### Issue: Port already in use

**Solution**: Change ports in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change 8000 to 8001
```

## 📈 Performance Optimization

### API Performance

- **Model Caching**: Model is loaded once and cached in memory
- **Async Endpoints**: FastAPI uses async for better concurrency
- **Connection Pooling**: MLflow client reuses connections

### Expected Performance

- **Latency P95**: < 1 second
- **Throughput**: 100+ requests/second (single container)
- **Inference Time**: ~10-50ms (depends on model complexity)

## 🔒 Security Best Practices

- ✅ Non-root user in Docker container
- ✅ Multi-stage Docker build for smaller images
- ✅ Health checks for all services
- ✅ Input validation with Pydantic
- ✅ Structured logging
- ✅ No hardcoded secrets (use environment variables)

## 🚀 Deployment Options

### Docker Compose (Development/Single Server)

```bash
docker-compose up -d
```

### Kubernetes (Production)

Convert docker-compose to Kubernetes manifests:
```bash
kompose convert
kubectl apply -f .
```

### Cloud Platforms

- **AWS**: ECS, EKS, or App Runner
- **Azure**: Container Instances or AKS
- **GCP**: Cloud Run or GKE

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review MLflow and FastAPI documentation

---

**Built with ❤️ for MLOps**
