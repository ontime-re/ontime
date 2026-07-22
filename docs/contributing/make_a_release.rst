.. image:: https://github.com/ontime-re/ontime/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/ontime-re/ontime/actions/workflows/ci.yml
   :alt: Continuous Integration


.. image:: https://badge.fury.io/py/ontime.svg
   :target: https://badge.fury.io/py/ontime
   :alt: PyPI version

==============
Make a Release
==============

This guide walks you through publishing a new version of onTime on PyPI.

#. Switch to the `main` branch

    .. code-block:: bash

        # make sure you are on the right branch
        git checkout main

#. Merge `develop` into `main`

    .. code-block:: bash

        git merge develop
        git push

#. Update the version in `pyproject.toml`

    .. code-block:: bash

        [project]
        name = "ontime"
        version = "x.y.z-suffix"

#. Commit and push

    .. code-block:: bash

        git add pyproject.toml
        git commit -m 'Update version to x.y.z-suffix'

#. Tag the version

    .. code-block:: bash

        git tag -a v<x.y.z-suffix> -m 'Version x.y.z-suffix'
        git push origin v<x.y.z-suffix>

#. Build the package

    .. code-block:: bash

        make build

    This runs ``uv build`` under the hood.

#. Publish the package

    .. code-block:: bash

        make publish

    This runs ``uv publish``. Then create the `GitHub Release <https://github.com/ontime-re/ontime/releases/new>`_.

#. Double-check everything went well

    * On `GitHub Actions <https://github.com/ontime-re/ontime/actions>`_.
    * On `PyPI <https://pypi.org/project/ontime/>`_.
    * Done! 🎉

