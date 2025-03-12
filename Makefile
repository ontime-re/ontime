
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
	make install-dependencies-docs

install-dependencies:
	@echo "Installing..."
	poetry install

install-dependencies-test:
	@echo "Installing..."
	poetry lock
	poetry install --with test

install-dependencies-docs:
	@echo "Installing..."
	sudo apt install -y pandoc
	poetry lock
	poetry install --with docs

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

run-notebooks:
	@echo "Running notebooks"
	poetry run python docs/run_notebooks.py

test:
	@echo "Running tests"
	poetry run pytest ./src/tests --disable-warnings

build:
	@echo "Building package"
	poetry build

build-docs:
	@echo "Building docs"
	poetry run m2r README.md --overwrite
	cd ./docs && poetry run make html

dev-docs:
	@echo "Start a development server for the documentation"
	sphinx-autobuild docs docs/_build/html

publish:
	@echo "Publishing package"
	poetry publish

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} \;
	rm -rf .pytest_cache

## must be run at ./ontime
check-notebooks:
	@echo "Checking notebooks"
	poetry run pytest --nbmake -n=auto notebooks

