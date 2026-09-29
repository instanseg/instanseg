# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "InstanSeg"
copyright = "2024, Thibaut Goldsborough, The University of Edinburgh"
author = "Thibaut Goldsborough"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    # Some docstrings use Google-style "Args:" sections rather than reST fields
    "sphinx.ext.napoleon",
]

exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# Strip the prompt from copied shell snippets
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# -- AutoAPI -----------------------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html
# AutoAPI parses the source instead of importing it, so none of the package's
# dependencies need to be installed to build the docs. Pages aren't generated
# automatically: the public API is listed by hand in docs/api/ so internal
# training code stays out of the reference.

autoapi_dirs = [str(ROOT / "instanseg")]
autoapi_ignore = ["*/scripts/*"]
autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
autoapi_options = ["members", "imported-members"]
# Both the class' and the __init__ method's docstring are concatenated and inserted
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
# The autoapi directives used in docs/api/ are autodoc documenters, so they follow autodoc settings
autodoc_member_order = "bysource"

# Show type hints next to each parameter rather than in long signatures
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
python_use_unqualified_type_names = True
# Page titles already name the module, so drop it from each signature
add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"InstanSeg {release}"
html_logo = str(ROOT / "assets" / "instanseg_logo.png")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/instanseg/instanseg",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/instanseg/instanseg",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15"
                "-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 "
                "2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 "
                '1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}


def _skip_attributes(app, what, name, obj, skip, options):
    """Hide instance attributes: the reference documents methods and functions only."""
    return True if getattr(obj, "type", None) == "attribute" else skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_attributes)
