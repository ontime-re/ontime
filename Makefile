
.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install-dependency-manager:
	@echo "Setting up..."
	export POETRY_HOME=/opt/poetry
	python -m venv $POETRY_HOME
	$POETRY_HOME/bin/pip install poetry
	$POETRY_HOME/bin/poetry --version

install-dependencies:
	@echo "Installing..."
	$POETRY_HOME/bin/poetry install
	$POETRY_HOME/bin/poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	$POETRY_HOME/bin/poetry shell

format:
	@echo "Formatting code"
	$POETRY_HOME/bin/poetry run black src/ontime

format-check:
	@echo "Checking code formatting"
	$POETRY_HOME/bin/poetry run black src/ontime --check

test:
	@echo "Running tests"
	$POETRY_HOME/bin/poetry run python -m unittest

build:
	@echo "Building package"
	$POETRY_HOME/bin/poetry build

publish:
	@echo "Publishing package"
	$POETRY_HOME/bin/poetry publish

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache