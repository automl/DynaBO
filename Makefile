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

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
ruff: ## run ruff as a formatter
	python -m ruff --exit-zero dynabo
	python -m ruff --silent --exit-zero --no-cache --fix dynabo
isort:
	python -m isort dynabo tests

test: ## run tests quickly with the default Python
	python -m pytest tests
cov-report:
	coverage html -d coverage_html

coverage: ## check code coverage quickly with the default Python
	coverage run --source dynabo -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/dynabo.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ dynabo
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html
install: clean ## install the package to the active Python's site-packages
	pip install -e ".[dev]"
	pip install -r requirements.txt
	git clone --branch development https://github.com/automl/SMAC3.git lib/SMAC3
	pip install -e lib/SMAC3
	git clone https://github.com/benjamc/yahpo_gym.git lib/yahpo_gym
	pip install -e lib/yahpo_gym/yahpo_gym
	python scripts/patch_yahpo_configspace.py
	pip install ConfigSpace --upgrade
	cd CARP-S 
	make benchmark_mfpbench
	pip install -e CARP-S

check:
	pre-commit run --all-files

format:
	make ruff
	make isort