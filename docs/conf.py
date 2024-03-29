# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Check the theme options here: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/branding.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/ontime"))
print(sys.path[0])

# -- Project information -----------------------------------------------------

project = "onTime"
copyright = "2024, onTime"
author = "onTime"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "recommonmark",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "./css/custom.css",
]

# -- NB Sphinx configuration --------------------------------------------------

nbsphinx_allow_errors = True

# -- Autosummary configuration ------------------------------------------------

autosummary_generate = True

# -- Pydata Sphinx Theme configuration ----------------------------------------

html_theme_options = {
    "github_url": "https://github.com/ontime-re/ontime",
    "external_links": [
        {"name": "Releases", "url": "https://pypi.org/project/ontime/#history"},
    ],
    "show_prev_next": True,
    "logo": {
        "alt_text": "onTime Python Package Documentation",
        "link": "https://ontime.re",
    },
    "logo": {
        "image_light": "ontime-logo.png",
        "image_dark": "ontime-logo_wh.png",
    },
}
