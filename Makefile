.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install check format
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

ruff: ## run ruff as a formatter
	python -m ruff --exit-zero dynabo
	python -m ruff --silent --exit-zero --no-cache --fix dynabo
isort:
	python -m isort dynabo tests

test: ## run tests quickly with the default Python
	python -m pytest tests
install: clean ## install the package to the active Python's site-packages
	uv sync
	uv run python scripts/patch_yahpo_configspace.py
	cd CARP-S/
	uv sync
	make benchmark_mfpbench
	cd ..
check:
	pre-commit run --all-files

format:
	make ruff
	make isort