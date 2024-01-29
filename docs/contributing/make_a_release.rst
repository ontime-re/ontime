.. image:: https://github.com/fredmontet/ontime/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/fredmontet/ontime/actions/workflows/ci.yml
   :alt: Continuous Integration


.. image:: https://badge.fury.io/py/ontime.svg
   :target: https://badge.fury.io/py/ontime
   :alt: PyPI version

==============
Make a Release
==============

This is a guides to publish onTime on PyPI.

#. Change branch to `main`

    .. code-block:: bash

        # make sure you are on the right branch
        git checkout main

#. Merge `develop` on `main`

    .. code-block:: bash

        git merge develop
        git push

#. Update the version in `pyproject.toml`

    .. code-block:: bash

        [tool.poetry]
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

#. Publish the package

    .. code-block:: bash

        make publish

    Also, make the `GitHub Release <https://github.com/fredmontet/ontime/releases/new>`_.

#. Check if everything went well

    * On `GitHub Actions <https://github.com/fredmontet/ontime/actions>`_.
    * On `PyPI <https://pypi.org/project/ontime/>`_.
    * Done! 🎉
    
