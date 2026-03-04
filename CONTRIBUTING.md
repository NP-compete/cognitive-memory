# Contributing to Cognitive Memory

Thank you for your interest in contributing to Cognitive Memory! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker (for running integration tests)
- Git

### Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/cognitive-memory.git
cd cognitive-memory
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:

```bash
make install-dev
```

4. Set up pre-commit hooks:

```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
make test-cov
```

### Code Style

We use:
- **Ruff** for linting and formatting
- **MyPy** for type checking

Run linters:

```bash
make lint
```

Auto-format code:

```bash
make format
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/NP-compete/cognitive-memory/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue with the `enhancement` label
3. Describe:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternatives you've considered

### Pull Requests

1. Create a branch from `main`:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our code style

3. Add tests for new functionality

4. Update documentation if needed

5. Run the test suite:

```bash
make test
make lint
```

6. Commit with a clear message:

```bash
git commit -m "feat: add support for custom decay functions"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

7. Push and create a Pull Request

### PR Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation as needed
- Ensure CI passes
- Request review from maintainers

## Architecture Overview

```
src/cognitive_memory/
├── core/           # Data models, config, exceptions
├── engines/        # Core algorithms (decay, importance, retrieval, consolidation)
├── tiers/          # Memory tier implementations
├── storage/        # Storage backend adapters
├── integrations/   # LangGraph, LangChain integrations
├── workers/        # Background workers
├── api/            # REST API
└── manager.py      # Main orchestrator
```

### Key Components

- **DecayEngine**: Computes memory strength decay
- **ImportanceEngine**: Scores memory importance
- **RetrievalEngine**: Decay-aware retrieval with MMR
- **ConsolidationEngine**: Episodic → semantic transformation
- **ContextBuilder**: Assembles optimal LLM context

## Testing Guidelines

### Unit Tests

- Test individual functions/methods in isolation
- Mock external dependencies
- Use `pytest` fixtures for setup

### Integration Tests

- Test interactions between components
- Require Docker for external services
- Mark with `@pytest.mark.integration`

### Property-Based Tests

- Use `hypothesis` for property testing
- Test invariants (e.g., decay is monotonic)

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Update docs/ for architectural changes

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a git tag
4. GitHub Actions publishes to PyPI

## Questions?

- Open a [Discussion](https://github.com/NP-compete/cognitive-memory/discussions)
- Check existing issues and PRs

Thank you for contributing!
