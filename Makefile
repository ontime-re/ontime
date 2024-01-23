
.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

get-informations:
	@echo "Python path"
	python -c "import sys; print('\n'.join(sys.path))"

	@echo "Versions"
	@echo "Python version: $(shell python --version)"
	@echo "Poetry version: $(shell poetry --version)"
	@echo "Pre-commit version: $(shell pre-commit --version)"

post-create-dev-container:
	@echo "Get container ready"
	@echo "› Installing dependencies with test"
	make install-dependencies-test

install-dependencies:
	@echo "Installing..."
	poetry install

install-dependencies-test:
	@echo "Installing..."
	poetry lock --no-update
	poetry install --with test

activate:
	@echo "Activating virtual environment"
	poetry shell

format:
	@echo "Formatting codebase"
	poetry run black src

format-check:
	@echo "Checking code formatting"
	poetry run black src --check

jupyter:
	@echo "Running Jupyter Lab"
	poetry run jupyter-lab .

test:
	@echo "Running tests"
	poetry run pytest ./src/tests --disable-warnings

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

