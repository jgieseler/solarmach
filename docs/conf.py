# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib import metadata
project = 'solarmach'
copyright = '2023, Jan Gieseler'
author = 'Jan Gieseler'
release = metadata.version(project)
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx',
              'nbsphinx_link',
              'sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx_automodapi.automodapi',
              'sphinx_automodapi.smart_resolver',
              'sphinx_copybutton',
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

napoleon_google_docstring = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
