import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Literal
import toml
from sphinx_gallery.sorting import FileNameSortKey

# Setup paths
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))

# Project Information from pyproject.toml
pyproject_data = toml.load(base_dir / "pyproject.toml")
project_info = pyproject_data["project"]

project = project_info["name"]
author = ", ".join(
    [
        f"{a.get('name', '')} ({a.get('email', '')})"
        for a in project_info.get("authors", [])
    ]
)
release = version = project_info["version"]
copyright = (
    f"2025 - {datetime.now().year}, n-squared lab, FAU Erlangen-Nürnberg, Germany"
)

# Import the main package
import myogen

# Sphinx Configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_design",
]

mermaid_version = "11.9.0"

napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_references = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_include_init_with_doc = False
napoleon_type_aliases = {
    # Standard typing aliases for cleaner docstrings
    "optional": "Optional[Any]",
    "array_like": ":term:`array_like <numpy:array_like>`",
    "dict_like": ":term:`dict-like <mapping>`",
    # NumPy and scientific computing types
    "ndarray": ":class:`numpy.ndarray`",
    "np.ndarray": ":class:`numpy.ndarray`",
    "float_array": ":class:`numpy.ndarray`\\[float]",
    "bool_array": ":class:`numpy.ndarray`\\[bool]",
    "int_array": ":class:`numpy.ndarray`\\[int]",
    "list[np.ndarray]": ":class:`list`\\[:class:`numpy.ndarray`]",
    "list[ndarray]": ":class:`list`\\[:class:`numpy.ndarray`]",
    "dtype[bool]": ":class:`numpy.dtype`\\[bool]",
    "tuple[int, ...]": ":class:`tuple`\\[int, ...]",
    "ndarray[tuple[int, ...], dtype[bool]]": ":class:`numpy.ndarray`\\[bool]",
    "float | list[float]": ":class:`float` | :class:`list`\\[:class:`float`]",
    "int | list[int]": ":class:`int` | :class:`list`\\[:class:`int`]",
    "str | list[str]": ":class:`str` | :class:`list`\\[:class:`str`]",
    "list[int] | None": ":class:`list`\\[:class:`int`] | :class:`None`",
    "list[float] | None": ":class:`list`\\[:class:`float`] | :class:`None`",
    "list[str] | None": ":class:`list`\\[:class:`str`] | :class:`None`",
    "tuple[int, int]": ":class:`tuple`\\[:class:`int`, :class:`int`]",
    "tuple[float, float]": ":class:`tuple`\\[:class:`float`, :class:`float`]",
    "list[int]": ":class:`list`\\[:class:`int`]",
    "list[float]": ":class:`list`\\[:class:`float`]",
    "list[str]": ":class:`list`\\[:class:`str`]",
    # MyoGen custom types - link to documentation
    "INPUT_CURRENT__MATRIX": ":data:`~myogen.utils.types.INPUT_CURRENT__MATRIX`",
    "SPIKE_TRAIN__MATRIX": ":data:`~myogen.utils.types.SPIKE_TRAIN__MATRIX`",
    "MUAP_SHAPE__TENSOR": ":data:`~myogen.utils.types.MUAP_SHAPE__TENSOR`",
    "SURFACE_EMG__TENSOR": ":data:`~myogen.utils.types.SURFACE_EMG__TENSOR`",
    "INTRAMUSCULAR_EMG__TENSOR": ":data:`~myogen.utils.types.INTRAMUSCULAR_EMG__TENSOR`",
    "CORTICAL_INPUT__MATRIX": ":data:`~myogen.utils.types.CORTICAL_INPUT__MATRIX`",
    "SURFACE_MUAP_SHAPE__TENSOR": ":data:`~myogen.utils.types.SURFACE_MUAP_SHAPE__TENSOR`",
    "INTRAMUSCULAR_MUAP_SHAPE__TENSOR": ":data:`~myogen.utils.types.INTRAMUSCULAR_MUAP_SHAPE__TENSOR`",
    "INPUT_CURRENT__MATRIX | None": ":class:`~myogen.utils.types.INPUT_CURRENT__MATRIX` | :class:`None`",
    # Beartype and Annotated type patterns - map to clean aliases
    "Annotated[ndarray[tuple[int, ...], dtype[bool]], beartype.vale.Is[lambda x: x.ndim == 3]]": ":data:`~myogen.utils.types.SPIKE_TRAIN__MATRIX`",
    "Annotated[npt.NDArray[np.bool_], Is[lambda x: x.ndim == 3]]": ":data:`~myogen.utils.types.SPIKE_TRAIN__MATRIX`",
    "Annotated[npt.NDArray[np.floating], Is[lambda x: x.ndim == 2]]": ":data:`~myogen.utils.types.INPUT_CURRENT__MATRIX`",
    "Annotated[npt.NDArray[np.floating], Is[lambda x: x.ndim == 5]]": ":data:`~myogen.utils.types.SURFACE_EMG__TENSOR`",
    # Matplotlib types
    "Axes": ":class:`matplotlib.axes.Axes`",
    "Figure": ":class:`matplotlib.figure.Figure`",
    "Axes3D": ":class:`mpl_toolkits.mplot3d.axes3d.Axes3D`",
    "IterableType[Axes]": ":class:`beartype.cave.IterableType`\\[:class:`matplotlib.axes.Axes`]",
    # Beartype types
    "IterableType": ":class:`beartype.cave.IterableType`",
    # NeuroML types
    "Segment": ":class:`neuroml.Segment`",
    "list[Segment]": ":class:`list`\\[:class:`neuroml.Segment`]",
    "list[neo.core.segment.Segment]": ":class:`list`\\[:class:`neo.core.segment.Segment`]",
    # Common union types
    "str_or_path": "str | :class:`pathlib.Path`",
    "float_or_list": "float | list[float]",
    "int_or_list": "int | list[int]",
    "str_or_list": "str | list[str]",
    # Motor unit recruitment model literals
    "fuglevand": "``'fuglevand'``",
    "deluca": "``'deluca'``",
    "konstantin": "``'konstantin'``",
    "combined": "``'combined'``",
    "RecruitmentMode": "``'fuglevand'`` | ``'deluca'`` | ``'konstantin'`` | ``'combined'``",
    "WhatToRecord": ":class:`list`\\[:class:`dict`\\[``'variables'``, ``'to_file'``, ``'sampling_interval'``, ``'locations'``\\], :class:`Any`\\]",
    "ElectrodeGridDimensions": ":class:`tuple`\\[:class:`int`, :class:`int`]",
    "ElectrodeGridCenterPosition": ":class:`tuple`\\[:class:`float` | :class:`int`, :class:`float` | :class:`int`]",
    "list[ElectrodeGridCenterPosition]": ":class:`list`\\[:class:`tuple`\\[:class:`float` | :class:`int`, :class:`float` | :class:`int`\\]]",
    # Simulation and modeling types
    "Muscle": ":class:`~myogen.simulator.Muscle`",
    "MotorNeuronPool": ":class:`~myogen.simulator.MotorNeuronPool`",
    "SurfaceEMG": ":class:`~myogen.simulator.SurfaceEMG`",
}

# MyST-Parser configuration
myst_enable_extensions = [
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
myst_admonition_enable = True
myst_url_schemes = ["http", "https", "mailto", "ftp"]
myst_ref_domains = ["std"]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_inherit_docstrings = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autodoc_preserve_defaults = True
autodoc_typehints_format = "short"
autodoc_type_aliases = napoleon_type_aliases

# Better signature formatting
maximum_signature_line_length = 80
python_use_unqualified_type_names = True

# Advanced autodoc signature formatting
add_function_parentheses = True
add_module_names = False
show_authors = True

# Improved signature display
autodoc_signature_formatting = "multiline"
python_maximum_signature_line_length = 88

# Autosummary configuration
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False
autosummary_ignore_module_all = False

# General configuration
templates_path = ["templates"]
exclude_patterns = ["Thumbs.db", ".DS_Store"]

pygments_dark_style = "monokai"

# HTML theme configuration
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": f"https://github.com/NSquaredLab/{project}",
    "navbar_start": ["navbar-logo", "navbar-version.html", "header-text.html"],
    "show_prev_next": False,
    "navbar_align": "left",
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    # Enhanced navigation
    "use_edit_page_button": True,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navigation_depth": 4,
    # Search improvements
    "search_bar_text": "Search MyoGen docs...",
    # API documentation improvements
    "show_nav_level": 2,
    "collapse_navigation": False,
    # Pygments (syntax highlighting) configuration
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    # Header and footer customization
    "header_links_before_dropdown": 4,
    # Announcement bar
    "announcement": "⚠️ MyoGen is under active development. API may change.",
}

html_static_path = ["_static"]
html_logo = "_static/myogen_logo.png"
html_css_files = ["custom.css"]
html_title = f"{project} {version} Documentation"

# HTML context
html_context = {
    "AUTHOR": author,
    "VERSION": version,
    "DESCRIPTION": project_info.get("description", ""),
    "github_user": "NSquaredLab",  # Update with your GitHub username
    "github_repo": project,
    "github_version": "main",
    "doc_path": "docs",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "neuroml": ("https://neuroml.readthedocs.io/en/stable/", None),
    "neo": ("https://neo.readthedocs.io/en/latest/", None),
}

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": str(base_dir / "examples"),
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py",
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
    "show_memory": False,
    "plot_gallery": True,
    "download_all_examples": False,
    "first_notebook_cell": "%matplotlib inline",
}

# Warning suppressions
suppress_warnings = [
    "config.cache",
    "ref.citation",
]


def setup(app):
    """Setup function for custom configurations."""
    app.add_css_file("custom.css")
    app.add_js_file("custom.js")
