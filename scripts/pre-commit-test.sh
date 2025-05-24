#!/bin/bash

set -e

echo "Running linting..."
flake8 core/ config/ tests/

echo "Running type checks..."
mypy core/ config/

echo "Running unit tests..."
pytest tests/unit/ --cov=core --cov=config --cov-fail-under=80
