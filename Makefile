# Uses the project virtualenv when present (the environment the results were produced with).
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: test test-fast

# Full suite (pytest; torch-dependent tests are skipped if torch is missing).
test:
	PYTHONPATH=src $(PYTHON) -m pytest -q tests

# Stop at the first failure.
test-fast:
	PYTHONPATH=src $(PYTHON) -m pytest -q -x tests
