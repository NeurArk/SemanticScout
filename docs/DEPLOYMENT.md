# SemanticScout - Deployment Guide

**Version**: 1.0  
**Date**: May 2025  
**Status**: Production Ready

## 🚀 Deployment Overview

SemanticScout supports multiple deployment strategies from local development to cloud production. This guide covers all deployment scenarios with detailed instructions, configuration options, and best practices.

## 🏠 Local Development Deployment

### Prerequisites

```bash
# Required software
- Python 3.11+ (recommended 3.12)
- Git
- 4GB+ RAM available
- 10GB+ free disk space

# Optional but recommended
- Docker Desktop
- VS Code with Python extension
```

### Quick Start Setup

```bash
# 1. Clone repository
git clone https://github.com/neurark/SemanticScout.git
cd SemanticScout

# 2. Create virtual environment
python -m venv semantic_scout_env
source semantic_scout_env/bin/activate  # Linux/Mac
# or
semantic_scout_env\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env with your OpenAI API key

# 5. Initialize database
python scripts/setup.py

# 6. Start application
python app.py
```

### Environment Configuration

```bash
# .env file for local development
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Application settings
APP_NAME=SemanticScout
APP_VERSION=1.0.0
DEBUG=true
LOG_LEVEL=DEBUG

# Storage configuration
CHROMA_PERSIST_DIR=./data/chroma_db
UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE=104857600  # 100MB in bytes
SUPPORTED_FORMATS=pdf,docx,txt,md

# UI configuration
GRADIO_THEME=soft
GRADIO_SHARE=false
GRADIO_PORT=7860
GRADIO_HOST=127.0.0.1
```

### Directory Structure Setup

```bash
# Create required directories
mkdir -p data/{uploads,chroma_db,cache,logs}
mkdir -p assets/{images,styles,icons}

# Set permissions (Linux/Mac)
chmod 755 data/
chmod 777 data/uploads/
chmod 755 data/chroma_db/
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/{uploads,chroma_db,cache,logs} && \
    chmod 755 data/ && \
    chmod 777 data/uploads/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run application
CMD ["python", "app.py"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  semanticscout:
    build: .
    container_name: semanticscout-app
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=gpt-4.1
      - OPENAI_EMBEDDING_MODEL=text-embedding-3-large
      - GRADIO_HOST=0.0.0.0
      - GRADIO_PORT=7860
      - DEBUG=false
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    container_name: semanticscout-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### Docker Commands

```bash
# Build and start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f semanticscout

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Shell access
docker-compose exec semanticscout bash
```

## ☁️ Cloud Deployment Options

### Hugging Face Spaces (Recommended for Demo)

#### Setup Steps

1. **Create Space**:

   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as SDK
   - Select appropriate hardware (CPU for demo, GPU for production)

2. **Configuration Files**:

**requirements.txt**

```txt
gradio
langchain
langchain-community
langchain-openai
langgraph
openai
chromadb
unstructured[all-docs]
PyMuPDF
plotly
networkx
umap-learn
scikit-learn
python-dotenv
pydantic
```

**README.md for Spaces**

```markdown
---
title: SemanticScout
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# SemanticScout

Intelligent Document Search & Analysis powered by AI.

## Configuration

Add your OpenAI API key in the Secrets section:

- Key: `OPENAI_API_KEY`
- Value: `sk-your-api-key-here`
```

#### Environment Variables in Spaces

```bash
# Set in Spaces Settings > Repository secrets
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4.1
GRADIO_SHARE=true
```

### Railway Deployment

#### railway.json

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "healthcheckPath": "/",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### Deployment Steps

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Set environment variables
railway variables set OPENAI_API_KEY=sk-your-key-here
railway variables set OPENAI_MODEL=gpt-4.1

# Deploy
railway up
```

### Render Deployment

#### render.yaml

```yaml
services:
  - type: web
    name: semanticscout
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: OPENAI_MODEL
        value: gpt-4.1
      - key: GRADIO_HOST
        value: 0.0.0.0
      - key: GRADIO_PORT
        value: 10000
    healthCheckPath: /
```

### AWS Deployment

#### EC2 with Docker

```bash
# Launch EC2 instance (t3.medium recommended)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy application
git clone https://github.com/neurark/SemanticScout.git
cd SemanticScout
docker-compose up -d
```

#### ECS Fargate Task Definition

```json
{
  "family": "semanticscout-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "semanticscout",
      "image": "your-repo/semanticscout:latest",
      "portMappings": [
        {
          "containerPort": 7860,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GRADIO_HOST",
          "value": "0.0.0.0"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/semanticscout",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## 📊 Production Configuration

### Environment Variables for Production

```bash
# Application
APP_NAME=SemanticScout
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
OPENAI_API_KEY=sk-production-key-here
OPENAI_MODEL=gpt-4.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_MAX_RETRIES=3
OPENAI_TIMEOUT=60

# Database Configuration
CHROMA_PERSIST_DIR=/app/data/chroma_db
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Storage Configuration
UPLOAD_DIR=/app/data/uploads
MAX_FILE_SIZE=104857600
CLEANUP_INTERVAL=3600
BACKUP_ENABLED=true
BACKUP_INTERVAL=86400

# Security
ALLOWED_ORIGINS=https://yourdomain.com
API_RATE_LIMIT=100
API_BURST_LIMIT=200
SESSION_TIMEOUT=3600

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_ENABLED=true
PERFORMANCE_MONITORING=true
```

### Production Docker Compose

```yaml
version: "3.8"

services:
  semanticscout:
    image: semanticscout:latest
    container_name: semanticscout-prod
    ports:
      - "80:7860"
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - LOG_LEVEL=INFO
    env_file:
      - .env.production
    volumes:
      - app_data:/app/data
      - app_logs:/app/logs
    restart: always
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G

  nginx:
    image: nginx:alpine
    container_name: semanticscout-nginx
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - semanticscout
    restart: always

  monitoring:
    image: prom/prometheus
    container_name: semanticscout-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: always

volumes:
  app_data:
  app_logs:
  prometheus_data:
```

### Nginx Configuration

```nginx
events {
    worker_connections 1024;
}

http {
    upstream semanticscout {
        server semanticscout:7860;
    }

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        client_max_body_size 100M;

        location / {
            proxy_pass http://semanticscout;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support for Gradio
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }
}
```

## 🔧 Configuration Management

### Settings Hierarchy

```python
# config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    app_name: str = "SemanticScout"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_max_retries: int = 3
    openai_timeout: int = 60

    # Storage
    chroma_persist_dir: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"
    max_file_size: int = 104857600  # 100MB

    # UI
    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7860
    gradio_share: bool = False
    gradio_theme: str = "soft"

    # Security
    allowed_origins: Optional[str] = None
    api_rate_limit: int = 100
    session_timeout: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
settings = Settings()
```

### Logging Configuration

```python
# config/logging.py
import logging
import logging.config
from datetime import datetime

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'data/logs/semanticscout.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'json',
            'filename': 'data/logs/errors.log',
            'maxBytes': 10485760,
            'backupCount': 3
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'error': {
            'handlers': ['error_file'],
            'level': 'ERROR',
            'propagate': False
        }
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
```

## 📈 Monitoring and Health Checks

### Health Check Endpoint

```python
# health_check.py
from flask import Flask, jsonify
import psutil
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Comprehensive health check"""

    # Check disk space
    disk_usage = psutil.disk_usage('/')
    disk_free_gb = disk_usage.free / (1024**3)

    # Check memory
    memory = psutil.virtual_memory()
    memory_available_gb = memory.available / (1024**3)

    # Check API connectivity (mock)
    api_status = check_openai_api()

    # Check database
    db_status = check_chroma_db()

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': os.getenv('APP_VERSION', '1.0.0'),
        'checks': {
            'disk_space': {
                'status': 'ok' if disk_free_gb > 1 else 'warning',
                'free_gb': round(disk_free_gb, 2)
            },
            'memory': {
                'status': 'ok' if memory_available_gb > 0.5 else 'warning',
                'available_gb': round(memory_available_gb, 2),
                'percent_used': memory.percent
            },
            'openai_api': api_status,
            'database': db_status
        }
    }

    # Overall status
    if any(check['status'] == 'error' for check in health_status['checks'].values()):
        health_status['status'] = 'unhealthy'
        return jsonify(health_status), 503
    elif any(check['status'] == 'warning' for check in health_status['checks'].values()):
        health_status['status'] = 'degraded'
        return jsonify(health_status), 200

    return jsonify(health_status), 200

def check_openai_api():
    try:
        # Implement OpenAI API connectivity check
        return {'status': 'ok', 'response_time_ms': 150}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_chroma_db():
    try:
        # Implement ChromaDB connectivity check
        return {'status': 'ok', 'collections': 1}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
```

### Monitoring with Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "semanticscout"
    static_configs:
      - targets: ["semanticscout:7860"]
    metrics_path: "/metrics"
    scrape_interval: 30s

  - job_name: "node"
    static_configs:
      - targets: ["localhost:9100"]
```

## 🛠️ Maintenance and Updates

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/semanticscout"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup ChromaDB
tar -czf "$BACKUP_DIR/chroma_db_$DATE.tar.gz" data/chroma_db/

# Backup uploaded documents
tar -czf "$BACKUP_DIR/uploads_$DATE.tar.gz" data/uploads/

# Backup logs
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" data/logs/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### Update Procedure

```bash
#!/bin/bash
# update.sh

echo "Starting SemanticScout update..."

# 1. Backup current state
./scripts/backup.sh

# 2. Pull latest code
git fetch origin
git checkout main
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Run migrations if needed
python scripts/migrate.py

# 5. Restart application
docker-compose down
docker-compose up -d --build

# 6. Verify health
sleep 30
curl -f http://localhost:7860/health || echo "Health check failed!"

echo "Update completed!"
```

### Scaling Considerations

```yaml
# docker-compose.scale.yml
version: "3.8"

services:
  semanticscout:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  load_balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - semanticscout
```

---

_This deployment guide provides comprehensive instructions for all deployment scenarios, ensuring SemanticScout can be reliably deployed from development to production environments._
