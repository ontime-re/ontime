# CLAUDE.md

## Project Overview

**OnTime** is a Python time series library (v0.6) for anomaly detection, model forecasting, and benchmarking. It targets energy and district heating network domains.

- **Python:** 3.10–3.12
- **Package manager:** `uv`
- **Key dependencies:** Darts, PyTorch, TensorFlow, Lightning, CatBoost, Altair, skforecast

## Commands

```bash
# Install dependencies
make install-dependencies        # all groups
make install-dependencies-test   # test group only

# Format
make format                      # black formatting
make format-check                # verify formatting (CI check)

# Test
make test                        # uv run pytest ./src/tests --disable-warnings

# Build
make build                       # uv build
make build-docs                  # Sphinx docs

# Clean
make clean                       # remove __pycache__, .pytest_cache, etc.
```

## Project Structure

```
src/
  ontime/
    api/          # High-level user-facing API
    context/      # Domain-specific detectors and tools
    core/         # Core modules: time_series, detection, generation, modelling, plotting, processing
    module/       # Advanced modules: benchmarking, datasets, pytorch/tensorflow processing
  tests/          # Mirrors src/ontime/ structure
docs/             # Sphinx documentation + notebooks
```

## Testing Conventions

- Test files: `src/tests/` mirroring `src/ontime/`
- File naming: `test_<module_name>.py`
- Function naming: `test_<MethodName>__<StateUnderTest>__<ExpectedBehavior>()`
- Example: `test_constructor__creation_from_dataframe__should_create_object_with_correct_data()`

## Code Style

- Formatter: Black (enforced in CI via `make format-check`)
- Always run `make format` before committing
