
.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

get-informations:
	@echo "Python path"
	uv run python -c "import sys; print('\n'.join(sys.path))"

	@echo "Versions"
	@echo "Python version: $(shell uv run python --version)"
	@echo "Uv version: $(shell uv --version)"
	@echo "Pre-commit version: $(shell uv run pre-commit --version)"

post-create-dev-container:
	@echo "Get container ready"
	@echo "› Installing dependencies with test"
	make install-dependencies-test
	make install-dependencies-docs

upgrade-dependencies:
	@echo "Upgrading all dependencies..."
	uv lock --upgrade
	uv sync

install-dependencies:
	@echo "Installing..."
	uv lock
	uv sync

install-dependencies-test:
	@echo "Installing..."
	uv lock
	uv sync --group test

install-dependencies-docs:
	@echo "Installing..."
	sudo apt install -y pandoc
	uv lock
	uv install --group docs

activate:
	@echo "Activating virtual environment"
	. .venv/bin/activate

format:
	@echo "Formatting codebase"
	uv run black src

format-check:
	@echo "Checking code formatting"
	uv run black src --check

jupyter:
	@echo "Running Jupyter Lab"
	uv run jupyter-lab .

run-notebooks:
	@echo "Running notebooks"
	uv run docs/run_notebooks.py

test:
	@echo "Running tests"
	uv run pytest ./src/tests --disable-warnings

build:
	@echo "Building package"
	uv build

build-docs:
	@echo "Building docs"
	uv run m2r README.md --overwrite
	cd ./docs && uv run make html

dev-docs:
	@echo "Start a development server for the documentation"
	sphinx-autobuild docs docs/_build/html

publish:
	@echo "Publishing package"
	uv publish

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} \;
	rm -rf .pytest_cache

## must be run at ./ontime
check-notebooks:
	@echo "Checking notebooks"
	uv run pytest --nbmake -n=auto notebooks

