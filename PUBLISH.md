[![Continuous Integration](https://github.com/ontime-re/ontime/actions/workflows/ci.yml/badge.svg)](https://github.com/ontime-re/ontime/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/ontime.svg)](https://badge.fury.io/py/ontime)

Publish onTime on PyPI
======================

A quick guide to publishing a new version of onTime on PyPI.

> This file mirrors `docs/contributing/make_a_release.rst`, the version
> published on [ontime.re](https://ontime.re/contributing). Keep both in
> sync if you update the release process.

## Steps

Switch to the `main` branch:

    git checkout main

Merge `develop` into `main`:

    git merge develop
    git push

Update the version in `pyproject.toml`:

    [project]
    name = "ontime"
    version = "x.y.z-suffix"

Commit and push:

    git add pyproject.toml
    git commit -m 'Update version to x.y.z-suffix'

Tag the version:

    git tag -a v<x.y.z-suffix> -m 'Version x.y.z-suffix'
    git push origin v<x.y.z-suffix>

Build the package (runs `uv build`):

    make build

Publish the package (runs `uv publish`):

    make publish

Check that the package is available on PyPI:

    https://pypi.org/project/ontime/#history

Done! 🎉
