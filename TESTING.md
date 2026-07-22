Testing Guideline
=================

This document describes the testing guidelines for the project.

## Structure of the repository

Tests live in the `src/tests` folder and mirror the directory structure of
`src/ontime`. In other words, every file under `src/ontime` has a matching
test file at the same relative path under `src/tests`.

For instance, `src/ontime/core/time_series/binary_time_series.py` has a
corresponding test file at `src/tests/core/time_series/test_binary_time_series.py`.

## Naming convention

All test files must follow this naming convention:

    test_<module_name>.py

For instance, `test_binary_time_series.py`.

Individual tests must follow this naming convention:

    test_<MethodName>__<StateUnderTest>__<ExpectedBehavior>()

For instance:

    test_constructor__creation_from_dataframe__should_create_object_with_correct_data()

## Test execution

Tests run automatically via GitHub Actions as part of the CI/CD pipeline on the
`develop` and `main` branches. To run them manually, use the same command the
CI uses:

    make test

This runs the entire test suite in the `src/tests` folder.
