# Adaptive Traffic Flow Lab - Development Makefile
# Usage: make <target>

.PHONY: help install test lint format typecheck clean \
        infra-up infra-down infra-logs infra-restart \
        vision-run state-run control-run api-run \
        train-yolo export-onnx build-docker

# Default target
help:
	@echo "Adaptive Traffic Flow Lab - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install all dependencies with uv"
	@echo "  pre-commit     Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  test           Run unit tests"
	@echo "  test-all       Run all tests (unit + integration)"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with ruff"
	@echo "  typecheck      Run mypy type checker"
	@echo "  check          Run all checks (lint + format + typecheck + test)"
	@echo ""
	@echo "Infrastructure:"
	@echo "  infra-up       Start all infrastructure services"
	@echo "  infra-down     Stop all infrastructure services"
	@echo "  infra-logs     View infrastructure logs"
	@echo "  infra-restart  Restart infrastructure services"
	@echo "  infra-ps       List infrastructure containers"
	@echo ""
	@echo "Applications:"
	@echo "  vision-run     Run vision worker on sample video"
	@echo "  state-run      Run traffic state service"
	@echo "  control-run    Run control service"
	@echo "  api-run        Run HTTP API server"
	@echo ""
	@echo "ML/Training:"
	@echo "  train-yolo     Train YOLO model on custom data"
	@echo "  export-onnx    Export model to ONNX format"
	@echo "  tensorboard    Start TensorBoard"
	@echo "  mlflow         Start MLflow UI"
	@echo ""
	@echo "Docker:"
	@echo "  build-docker   Build all Docker images"
	@echo "  push-docker    Push Docker images to registry"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean build artifacts and caches"
	@echo "  reset-db       Reset database (DANGEROUS)"

# =============================================================================
# Setup
# =============================================================================

install:
	uv sync --all-extras --dev
	uv run pre-commit install

pre-commit:
	uv run pre-commit install

# =============================================================================
# Development
# =============================================================================

test:
	uv run pytest tests/unit -v --tb=short

test-all:
	uv run pytest tests/unit tests/integration -v --tb=short

test-e2e:
	uv run pytest tests/e2e -v --tb=short

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy packages apps configs

check: lint format typecheck test

# =============================================================================
# Infrastructure
# =============================================================================

infra-up:
	cd infra/docker && docker compose up -d

infra-down:
	cd infra/docker && docker compose down

infra-logs:
	cd infra/docker && docker compose logs -f

infra-restart:
	cd infra/docker && docker compose restart

infra-ps:
	cd infra/docker && docker compose ps

infra-build:
	cd infra/docker && docker compose build

# Start with simulation profile (includes SUMO)
infra-up-sim:
	cd infra/docker && docker compose --profile simulation up -d

# =============================================================================
# Applications (to be implemented in later phases)
# =============================================================================

vision-run:
	@echo "Vision worker not yet implemented. Run: uv run python -m apps.vision_worker"

state-run:
	@echo "State service not yet implemented. Run: uv run python -m apps.state_service"

control-run:
	@echo "Control service not yet implemented. Run: uv run python -m apps.control_service"

api-run:
	@echo "API server not yet implemented. Run: uv run python -m apps.api"

# =============================================================================
# ML/Training
# =============================================================================

train-yolo:
	@echo "Train YOLO on custom data:"
	@echo "  uv run yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640"

export-onnx:
	@echo "Export model to ONNX:"
	@echo "  uv run yolo export model=models/checkpoints/best.pt format=onnx opset=12"

tensorboard:
	uv run tensorboard --logdir models/checkpoints --port 6006

mlflow:
	uv run mlflow ui --port 5000

# =============================================================================
# Docker
# =============================================================================

build-docker:
	docker build -t traffic-lab/vision_worker:latest -f apps/vision_worker/Dockerfile .
	docker build -t traffic-lab/state_service:latest -f apps/state_service/Dockerfile .
	docker build -t traffic-lab/control_service:latest -f apps/control_service/Dockerfile .
	docker build -t traffic-lab/api:latest -f apps/api/Dockerfile .

push-docker:
	docker push traffic-lab/vision_worker:latest
	docker push traffic-lab/state_service:latest
	docker push traffic-lab/control_service:latest
	docker push traffic-lab/api:latest

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist .coverage coverage.xml htmlcov 2>/dev/null || true

reset-db:
	@echo "WARNING: This will delete all data in TimescaleDB and ClickHouse!"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd infra/docker && docker compose down -v timescaledb clickhouse; \
		docker volume rm infra_docker_timescaledb_data infra_docker_clickhouse_data 2>/dev/null || true; \
		docker compose up -d timescaledb clickhouse; \
		echo "Databases reset."; \
	else \
		echo "Cancelled."; \
	fi

# =============================================================================
# Google Colab / Free GPU helpers
# =============================================================================

colab-setup:
	@echo "Run this in a Google Colab notebook:"
	@echo ""
	@echo "!git clone https://github.com/your-org/adaptive-traffic-flow-lab.git"
	@echo "!cd adaptive-traffic-flow-lab && pip install -e '.[all]'"
	@echo ""
	@echo "Then enable GPU: Runtime -> Change runtime type -> T4 GPU"

colab-train:
	@echo "Train YOLO in Colab:"
	@echo "  !cd adaptive-traffic-flow-lab && yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0"