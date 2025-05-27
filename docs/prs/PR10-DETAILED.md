# PR10: Simple Demo Deployment

## Overview
Deploy the app for easy demo access. Keep it simple - HuggingFace Spaces or local Docker.

## Goal
Make the demo accessible with minimal setup complexity.

## Option 1: HuggingFace Spaces (Recommended)

### 1. Create `app.py` in root
Already done in PR7 - just ensure it's in the root directory.

### 2. Create `requirements.txt`
Already exists - verify all dependencies are listed.

### 3. Create `.env.example`
```bash
OPENAI_API_KEY=your-key-here
```

### 4. Create `README.md` for Space
```markdown
---
title: SemanticScout
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# SemanticScout - Chat with your Documents

Upload documents and ask questions using AI.

## Setup
1. Fork this Space
2. Add your OpenAI API key in Settings → Variables
3. Start chatting with your documents!
```

### 5. Deploy to HuggingFace
```bash
# Install huggingface-cli
pip install huggingface-hub

# Login
huggingface-cli login

# Create space
huggingface-cli repo create SemanticScout --type space --space_sdk gradio

# Push code
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/SemanticScout
git push space main
```

## Option 2: Simple Docker (Local Demos)

### Create `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create directories
RUN mkdir -p data/uploads data/chroma_db logs

# Expose port
EXPOSE 7860

# Run app
CMD ["python", "app.py"]
```

### Create `docker-compose.yml`
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
```

### Run Locally
```bash
# Build and run
docker-compose up --build

# Access at http://localhost:7860
```

## Sample Documents

Create `samples/` directory with demo PDFs:
- `samples/contract_example.pdf`
- `samples/research_paper.pdf`
- `samples/product_manual.pdf`

## Quick Start Guide

### For `README.md`
```markdown
# SemanticScout

Chat naturally with your documents using AI.

## Quick Start

### Option 1: Use Hosted Demo
Visit: [https://huggingface.co/spaces/YOUR_USERNAME/SemanticScout]

### Option 2: Run Locally
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/SemanticScout
cd SemanticScout

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run app
python app.py
```

## Usage
1. Upload PDF, DOCX, or TXT files
2. Ask questions about your documents
3. Get AI-powered answers with sources

## Demo Questions
- "What are the key terms in this contract?"
- "Summarize the main findings"
- "What are the payment conditions?"
```

## What We're NOT Doing

- ❌ Kubernetes deployments
- ❌ Production monitoring
- ❌ Auto-scaling
- ❌ CI/CD pipelines
- ❌ Database backups

## Success Criteria

- [ ] Demo accessible via link
- [ ] Uploads and chat work smoothly
- [ ] Clear instructions for API key
- [ ] Sample documents ready

Remember: This is for demos, not production. Keep deployment dead simple.