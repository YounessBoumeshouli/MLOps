# MLOps Quick Start Guide

## 🚀 Start the Platform

```bash
cd C:\Users\boume\Briefs\MLOps
docker-compose up -d
```

**Services**:
- MLflow: http://localhost:5001
- API: http://localhost:8001
- API Docs: http://localhost:8001/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 🤖 Train a Model

```bash
python src/train.py
```

## 🧪 Test the API

```bash
# Health check
curl http://localhost:8001/health

# Prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [1.5, -0.3, 2.1, 0.8, -1.2, 0.5, 1.8, -0.7, 2.3, 0.2, -1.5, 1.1, 0.9, -0.4, 2.0, 0.7, -1.8, 1.3, 0.6, -0.9]}"
```

## 📊 View Dashboards

1. **Grafana**: http://localhost:3000 → Dashboards → MLOps API Dashboard
2. **MLflow**: http://localhost:5001 → Models → ml_classifier

## 🛑 Stop Services

```bash
docker-compose down
```

## 🔄 Rebuild

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📝 Run Tests

```bash
pytest tests/ -v --cov=src
```

## 🎨 Format Code

```bash
black src/ tests/
flake8 src/ tests/ --max-line-length=100
```
