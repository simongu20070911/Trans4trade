PY := python
PIP := pip

.PHONY: install-dev format lint lint-fix nb-lint check

install-dev:
	$(PIP) install -U pip
	$(PIP) install -r requirements-dev.txt
	pre-commit install || true

format:
	black .
	isort .

lint:
	ruff check .

lint-fix:
	ruff check --fix .

nb-lint:
	nbqa ruff --line-length 100 --ignore E501 .

check:
	black --check .
	ruff check .

