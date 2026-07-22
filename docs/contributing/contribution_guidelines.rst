Contribution guidelines
=======================

We're happy to welcome you as a contributor to the onTime project. This guide
covers the contribution workflow along with a few guidelines to help you get
started.


Contributing to the code
------------------------

If you'd like to contribute code, please open an issue first, or assign
yourself to an existing one. Check the issue board here_ to see whether
what you have in mind is already being tracked.


Branching model
---------------

We use a fork/branch and pull request workflow: create a branch for your
change, push your commits, and open a pull request against ``develop`` once
it's ready for review.

Please follow this branch naming convention:

    <issue number>-<issue slug>

For example, if you're working on issue #1, name your branch ``1-add-readme``.
If you're on an issue page, GitHub gives you a button in the sidebar to create
a branch with the correct name automatically.

Once your pull request is approved and CI passes, it can be merged into
``develop``. Releases are cut from ``main`` — see :doc:`make_a_release`.


Submit bug reports
------------------

To submit a bug report, please include a comprehensive issue description that
lets us reproduce the error.

If the bug can be reproduced in a notebook, a link to a version running on a
tool like Binder_ is ideal.


Submit enhancements and feature requests
-----------------------------------------

To suggest an enhancement or a new feature, the usual way is to open an issue.
If you'd rather discuss it with us live, feel free to reach out.


.. _here: https://github.com/ontime-re/ontime/issues
.. _Binder: https://mybinder.org/
