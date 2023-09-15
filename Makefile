
.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install-dependencies:
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

format:
	@echo "Formatting code"
	poetry run black src/ontime

format-check:
	@echo "Checking code formatting"
	poetry run black src/ontime --check

test:
	@echo "Running tests"
	poetry run python -m unittest

build:
	@echo "Building package"
	poetry build

publish:
	@echo "Publishing package"
	poetry publish

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache