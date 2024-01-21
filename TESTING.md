Testing Guideline
=================

This document describes the testing guideline for the project.

## Structure of the repository

The tests are located in the `src/tests` folder and follow the same 
directory structure as the `src/ontime` folder. This means that each 
file in the `src/ontime` subfolders has a test file for the corresponding
file in the `src/tests`.

For instance, `src/ontime/core/time_series/binary_time_series.py` has a corresponding 
test file `src/tests/core/time_series/test_binary_time_series.py`.

## Naming convention

All tests file must follow the naming convention:

    test_<module_name>.py

For instance, `test_binary_time_series.py`.

Then, individual tests must follow the following naming convention:

    test_<MethodName>__<StateUnderTest>__<ExpectedBehavior>()

For instance: 
    
    `test_constructor__creation_from_dataframe__should_create_object_with_correct_data()`.

## Test execution

The tests are executed automatically with GitHub Actions within the CI/CD pipeline on the
`develop` and `main` branch. If you want to run the test manually, you can call the same 
command than the CI does locally by running the following command:
    
    make test

This will run all the tests in the `src/tests` folder.


