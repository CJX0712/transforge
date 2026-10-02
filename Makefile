# TransForge -- verified optimal-transport solver bench
# Author: 晨星 (CJX0712)
PYTHON ?= python
OUT    ?= benchmark.json

.DEFAULT_GOAL := help
.PHONY: help install lint format test coverage invariants bench demo reproduce docker clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:          ## install the package and dev dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:             ## ruff check + format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:           ## apply ruff formatting
	$(PYTHON) -m ruff format .

test:             ## run the test suite
	$(PYTHON) -m pytest -q -W ignore::UserWarning

coverage:         ## test suite with coverage
	$(PYTHON) -m pytest -q -W ignore::UserWarning --cov=transforge --cov-report=term-missing

invariants:       ## all ten cross-validation invariants (non-zero exit on failure)
	$(PYTHON) -m transforge.cli invariant --strict

bench:            ## full benchmark -> $(OUT)
	$(PYTHON) -m transforge.cli bench --seeds 3 --out $(OUT)

demo:             ## end-to-end demonstration -> $(OUT)
	$(PYTHON) examples/run_demo.py --out $(OUT)

reproduce:        ## run twice, verify bitwise determinism
	$(PYTHON) -m transforge.cli reproduce

check: lint test invariants   ## what CI runs

docker:           ## build the container image
	docker build -t transforge:local .

clean:            ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
