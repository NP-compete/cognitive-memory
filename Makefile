.PHONY: help install install-dev lint format test test-unit test-integration test-cov clean build publish docker-build docker-run

PYTHON := python3
PIP := pip

help:
	@echo "cognitive-memory - Memory that forgets, like humans do"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install package"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make clean          Clean build artifacts"
	@echo "  make build          Build package"
	@echo "  make publish        Publish to PyPI"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	mypy src/cognitive_memory

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "not integration"

test-integration:
	pytest tests/integration/ -v -m integration

test-cov:
	pytest tests/ --cov=src/cognitive_memory --cov-report=term-missing --cov-report=html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	$(PYTHON) -m build

publish: build
	$(PYTHON) -m twine upload dist/*

docker-build:
	docker build -t cognitive-memory:latest -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down
