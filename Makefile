
.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

get-informations:
	@echo "Python path"
	python -c "import sys; print('\n'.join(sys.path))"

	@echo "Versions"
	@echo "Python version: $(shell python --version)"
	@echo "Poetry version: $(shell poetry --version)"
	@echo "Pre-commit version: $(shell pre-commit --version)"

install-dependencies:
	@echo "Installing with option dev packages..."
	poetry install --with dev
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

format:
	@echo "Formatting codebase"
	poetry run black src/ontime

format-check:
	@echo "Checking code formatting"
	poetry run black src/ontime --check

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

## must be run at ./ontime
check-notebooks:
	@echo "Checking notebooks"
	poetry run pytest --nbmake -n=auto notebooks

