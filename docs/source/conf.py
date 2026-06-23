import sys
from datetime import datetime
from pathlib import Path

import toml
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey


def reset_neuron(gallery_conf, fname):
    """
    Reset NEURON state between Sphinx Gallery examples.

    NEURON's HOC interpreter maintains global state that persists across
    examples when they run in the same Python process. This causes examples
    to fail on first run but succeed on subsequent runs.
    """
    try:
        # Import myogen to ensure NEURON is properly initialized
        # (myogen auto-loads mechanisms and sets up NEURON on import)
        import myogen  # noqa: F401
        from neuron import h

        # Delete all sections to start fresh
        for sec in list(h.allsec()):
            h.delete_section(sec=sec)

        # Load stdrun.hoc and reset time variables
        h.load_file("stdrun.hoc")
        h.t = 0
        h.tstop = 0

    except (ImportError, RuntimeError, LookupError, AttributeError):
        pass  # NEURON not available or not initialized, skip reset

# Setup paths
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))

# Project Information from pyproject.toml
pyproject_data = toml.load(base_dir / "pyproject.toml")
project_info = pyproject_data["project"]

project = project_info["name"]
author = ", ".join(
    [f"{a.get('name', '')} ({a.get('email', '')})" for a in project_info.get("authors", [])]
)
release = version = project_info["version"]
copyright = f"2025 - {datetime.now().year}, n-squared lab, FAU Erlangen-Nürnberg, Germany"

# Import the main package
import myogen


# Copy README.md from root to docs/source with path fixes
def copy_and_prepare_readme():
    """Copy README.md to docs/source and append RST sections."""
    import re

    readme_src = base_dir / "README.md"
    index_rst = Path(__file__).parent / "index.rst"

    if readme_src.exists():
        content = readme_src.read_text(encoding="utf-8")

        # Fix image paths for docs context
        content = content.replace('src="./docs/source/_static/', 'src="_static/')
        content = content.replace('src="docs/source/_static/', 'src="_static/')

        # Fix documentation links
        content = content.replace("](docs/", "](")

        # Convert GitHub-style alerts to MyST admonitions
        # Pattern: > [!TYPE]\n> content
        def convert_gh_alert(match):
            alert_type = match.group(1).lower()
            alert_content = match.group(2).strip()
            # Remove leading > from subsequent lines
            alert_content = re.sub(r"^> ", "", alert_content, flags=re.MULTILINE)
            return f":::{{{alert_type}}}\n{alert_content}\n:::\n"

        # Match GitHub alert blocks: > [!TYPE] followed by lines starting with >
        content = re.sub(
            r"> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\n((?:> .+\n?)+)",
            convert_gh_alert,
            content,
            flags=re.IGNORECASE,
        )

        # Append package structure and toctrees as eval-rst block
        rst_appendix = """

```{eval-rst}
----

Package Structure
-----------------

.. code-block:: text

   MyoGen/
   ├── myogen/              # Main package source code
   │   ├── simulator/       # Core simulation functionality
   │   │   ├── core/        # Core simulation components
   │   │   │   ├── emg/     # EMG signal generation
   │   │   │   ├── muscle/  # Muscle modeling
   │   │   │   └── spike_train/ # Motor neuron simulation
   │   │   └── ...
   │   ├── utils/           # Utility functions and tools
   │   │   ├── plotting/    # Visualization utilities
   │   │   ├── currents.py  # Current generation
   │   │   └── nmodl.py     # NMODL file handling
   │   └── ...
   ├── examples/            # Example scripts and tutorials
   ├── docs/                # Documentation source
   ├── pyproject.toml       # Project metadata and dependencies
   └── uv.lock              # Pinned versions of dependencies



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Documentation

   api/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   neo_blocks_guide

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   :hidden:

   examples
```
"""
        content += rst_appendix

        # Save as index.md (which MyST will parse as the root document)
        index_md = Path(__file__).parent / "index.md"
        index_md.write_text(content, encoding="utf-8")
        print("✓ README.md integrated into index.md")
    else:
        print(f"⚠️ WARNING: README.md not found at {readme_src}")


# Run during module load
copy_and_prepare_readme()

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
    "hoverxref.extension",
    "sphinx_autodoc_typehints",  # Automatic type hint formatting and linking
]

mermaid_version = "11.9.0"

napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_references = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False  # Let sphinx-autodoc-typehints handle types
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
    # MyoGen Quantity types - link to type documentation
    "Quantity__s": ":data:`~myogen.utils.types.Quantity__s`",
    "Quantity__ms": ":data:`~myogen.utils.types.Quantity__ms`",
    "Quantity__rad": ":data:`~myogen.utils.types.Quantity__rad`",
    "Quantity__deg": ":data:`~myogen.utils.types.Quantity__deg`",
    "Quantity__mV": ":data:`~myogen.utils.types.Quantity__mV`",
    "Quantity__uV": ":data:`~myogen.utils.types.Quantity__uV`",
    "Quantity__nA": ":data:`~myogen.utils.types.Quantity__nA`",
    "Quantity__uS": ":data:`~myogen.utils.types.Quantity__uS`",
    "Quantity__S_per_m": ":data:`~myogen.utils.types.Quantity__S_per_m`",
    "Quantity__Hz": ":data:`~myogen.utils.types.Quantity__Hz`",
    "Quantity__pps": ":data:`~myogen.utils.types.Quantity__pps`",
    "Quantity__mm": ":data:`~myogen.utils.types.Quantity__mm`",
    "Quantity__m": ":data:`~myogen.utils.types.Quantity__m`",
    "Quantity__mm2": ":data:`~myogen.utils.types.Quantity__mm2`",
    "Quantity__per_mm2": ":data:`~myogen.utils.types.Quantity__per_mm2`",
    "Quantity__m_per_s": ":data:`~myogen.utils.types.Quantity__m_per_s`",
    "Quantity__mm_per_s": ":data:`~myogen.utils.types.Quantity__mm_per_s`",
    # Quantity tuple types
    "tuple[Quantity__m_per_s, Quantity__m_per_s]": ":class:`tuple`\\[:data:`~myogen.utils.types.Quantity__m_per_s`, :data:`~myogen.utils.types.Quantity__m_per_s`]",
    # AnalogSignal types
    "CURRENT__AnalogSignal": ":data:`~myogen.utils.types.CURRENT__AnalogSignal`",
    "INPUT_CURRENT__AnalogSignal": ":data:`~myogen.utils.types.CURRENT__AnalogSignal`",  # Alias to CURRENT__AnalogSignal
    "FORCE__AnalogSignal": ":data:`~myogen.utils.types.FORCE__AnalogSignal`",
    # MyoGen custom types - link to documentation
    "RECRUITMENT_THRESHOLDS__ARRAY": ":data:`~myogen.utils.types.RECRUITMENT_THRESHOLDS__ARRAY`",
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


# Custom type annotation formatters
def simplify_quantity_annotations(annotation):
    """Simplify Annotated[Quantity, ...] types to clean Quantity__* aliases."""
    import re

    annotation_str = str(annotation)

    # Map of unit patterns to type alias names
    quantity_patterns = {
        r"IsEqual\['s'\]": "Quantity__s",
        r"IsEqual\['ms'\]": "Quantity__ms",
        r"IsEqual\['rad'\]": "Quantity__rad",
        r"IsEqual\['deg'\]": "Quantity__deg",
        r"IsEqual\['mV'\]": "Quantity__mV",
        r"IsEqual\['uV'\]": "Quantity__uV",
        r"IsEqual\['nA'\]": "Quantity__nA",
        r"IsEqual\['uS'\]": "Quantity__uS",
        r"IsEqual\['S/m'\]": "Quantity__S_per_m",
        r"IsEqual\['Hz'\]": "Quantity__Hz",
        r"IsEqual\['pps'\]": "Quantity__pps",
        r"IsEqual\['mm'\]": "Quantity__mm",
        r"IsEqual\['m'\]": "Quantity__m",
        r"IsEqual\['mm\*\*2'\]": "Quantity__mm2",
        r"IsEqual\['1/mm\*\*2'\]": "Quantity__per_mm2",
        r"IsEqual\['m/s'\]": "Quantity__m_per_s",
        r"IsEqual\['mm/s'\]": "Quantity__mm_per_s",
    }

    # Try to match typing.Annotated[quantities.quantity.Quantity, beartype.vale.IsAttr[...]]
    # Pattern matches with or without module prefixes
    for unit_pattern, type_alias in quantity_patterns.items():
        pattern = rf"Annotated\[.*?Quantity,\s*.*?IsAttr\['dimensionality',\s*.*?IsAttr\['unicode',\s*.*?{unit_pattern}\]\]\]"
        if re.search(pattern, annotation_str):
            return type_alias

    return None


def format_annotation(annotation, config=None):
    """Custom formatter to simplify Quantity annotations.

    Parameters
    ----------
    annotation : Any
        The type annotation to format
    config : sphinx.config.Config, optional
        Sphinx configuration object (may not be provided in all contexts)

    Returns
    -------
    str or None
        Formatted reStructuredText for the annotation, or None to use default formatting
    """
    import typing

    # Try to simplify Quantity annotations
    simplified = simplify_quantity_annotations(annotation)
    if simplified:
        # Try importing the type alias and returning it directly
        try:
            from myogen.utils import types

            return getattr(types, simplified)
        except Exception:
            # If that fails, return RST reference
            return f":data:`~myogen.utils.types.{simplified}`"

    # Return None to let sphinx-autodoc-typehints handle it normally
    return None


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

# sphinx-autodoc-typehints configuration
typehints_use_rtype = True  # Show return types in :rtype: field
typehints_document_rtype = True  # Document return types
typehints_defaults = "comma"  # Show default values with commas
typehints_use_signature = True  # Put types in signature for better rendering
typehints_use_signature_return = True  # Put return type in signature
typehints_fully_qualified = False  # Use short names (not myogen.utils.types.Foo)
always_use_bars_union = True  # Use | instead of Union in docs (Python 3.10+ style)

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

# Syntax highlighting with Pygments
pygments_style = "monokai"  # Default/fallback style
pygments_dark_style = "monokai"  # Dark mode

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
    "use_edit_page_button": False,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navigation_depth": 4,
    # Search improvements
    "search_bar_text": "Search MyoGen docs...",
    # API documentation improvements
    "show_nav_level": 2,
    "collapse_navigation": False,
    # Pygments (syntax highlighting) configuration
    "pygments_light_style": "friendly",  # Clean light theme that pairs well with Monokai
    "pygments_dark_style": "monokai",
    # Header and footer customization
    "header_links_before_dropdown": 4,
    # Announcement bar
    "announcement": "MyoGen is under active development. API may change.",
}

html_static_path = ["_static"]
html_logo = "_static/myogen_logo.png"
html_css_files = ["custom.css"]
html_title = f"{project} {version} Documentation"
html_show_sourcelink = False

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
    "neo": ("https://neo.readthedocs.io/en/latest/", None),
    "beartype": ("https://beartype.readthedocs.io/en/latest/", None),
    "quantities": ("https://python-quantities.readthedocs.io/en/latest/", None),
}

# Sphinx-hoverxref configuration for hover previews
hoverxref_auto_ref = True  # Automatically add tooltips to all references
hoverxref_domains = ["py"]  # Enable for Python domain
hoverxref_roles = [
    "class",
    "func",
    "meth",
    "attr",
    "data",
    "mod",
    "obj",
]  # Roles to enable hover preview for
hoverxref_role_types = {
    "class": "tooltip",  # Use tooltip style for classes
    "func": "tooltip",  # Use tooltip style for functions
    "meth": "tooltip",  # Use tooltip style for methods
    "attr": "tooltip",  # Use tooltip style for attributes
    "data": "tooltip",  # Use tooltip style for data
    "mod": "modal",  # Use modal style for modules (larger preview)
    "ref": "tooltip",  # Use tooltip for references
    "obj": "tooltip",  # Use tooltip style for generic objects
}
hoverxref_tooltip_maxwidth = 600  # Maximum width of tooltip in pixels
hoverxref_tooltip_animation_duration = 200  # Animation duration in milliseconds
hoverxref_intersphinx = [
    "python",
    "numpy",
    "scipy",
    "matplotlib",
]  # Enable hover previews for intersphinx references

# Embed tooltip content for local viewing (fixes "Loading..." issue with file:// URLs)
hoverxref_tooltip_lazy = False  # Disable lazy loading to embed content directly in HTML

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": [
        str(base_dir / "examples" / "01_basic"),
        str(base_dir / "examples" / "02_finetune"),
        str(base_dir / "examples" / "03_papers" / "watanabe"),
    ],
    "gallery_dirs": [
        "auto_examples/01_basic",
        "auto_examples/02_finetune",
        "auto_examples/03_papers/watanabe",
    ],
    "subsection_order": ExplicitOrder(
        [
            str(base_dir / "examples" / "01_basic"),
            str(base_dir / "examples" / "02_finetune"),
            str(base_dir / "examples" / "03_papers" / "watanabe"),
        ]
    ),
    "filename_pattern": r"\.py",
    # 14_calibrate_noise_from_real.py needs a private iEMG .mat recording that
    # isn't available in CI, so it can't be executed by the gallery.
    "ignore_pattern": r"(14_calibrate_noise_from_real|_oscillating_dc_helpers|_optimize_dc_worker)\.py",
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
    "show_memory": False,
    "plot_gallery": True,
    "download_all_examples": False,
    "first_notebook_cell": "%matplotlib inline",
    "reset_modules": (reset_neuron,),  # Reset NEURON state between examples
}

# Warning suppressions
suppress_warnings = [
    "config.cache",
    "ref.citation",
]


def prettify_type_alias(_app, what, _name, _obj, _options, lines):
    """Pretty-print type alias beartype annotations."""
    import re

    # Only process data (type aliases)
    if what != "data":
        return

    # Check for beartype Annotated types and format them better
    for i, line in enumerate(lines):
        if "alias of" in line and "Annotated[" in line:
            # Extract the annotation content
            match = re.search(r"alias of (.+)", line)
            if match:
                annotation = match.group(1)
                # Format as a code block for better readability
                lines[i] = "**Type Alias:**\n\n.. code-block:: python\n\n   " + annotation.replace(
                    ", ", ",\n   "
                )
                lines.insert(i + 1, "")


def post_process_html(_app, exception):
    """Post-process HTML files to replace Annotated[Quantity, ...] with Quantity__* links."""
    if exception:
        return

    import re
    from pathlib import Path

    build_dir = Path(_app.outdir)

    # HTML replacements for Quantity types
    # Match specific Annotated[Quantity, IsAttr[...IsEqual['X']...]] patterns
    # Base pattern for all Quantity types
    base_pattern = r'<a class="hxr-hoverxref hxr-tooltip reference external" href="https://docs\.python\.org/3/library/typing\.html#typing\.Annotated"[^>]*><span class="pre">Annotated</span></a><span class="p"><span class="pre">\[</span></span><span class="pre">Quantity</span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a class="reference external" href="[^"]*beartype[^"]*IsAttr"[^>]*><span class="pre">IsAttr</span></a><span class="p"><span class="pre">\[</span></span><span class="s"><span class="pre">\'dimensionality\'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a class="reference external" href="[^"]*beartype[^"]*IsAttr"[^>]*><span class="pre">IsAttr</span></a><span class="p"><span class="pre">\[</span></span><span class="s"><span class="pre">\'unicode\'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a class="reference external" href="[^"]*beartype[^"]*IsEqual"[^>]*><span class="pre">IsEqual</span></a><span class="p"><span class="pre">\[</span></span><span class="s"><span class="pre">\'UNIT\'</span></span><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">\]</span></span>'

    # Map of unit strings to type alias names
    quantity_units = {
        "s": "Quantity__s",
        "ms": "Quantity__ms",
        "rad": "Quantity__rad",
        "deg": "Quantity__deg",
        "mV": "Quantity__mV",
        "uV": "Quantity__uV",
        "nA": "Quantity__nA",
        "uS": "Quantity__uS",
        "S/m": "Quantity__S_per_m",
        "Hz": "Quantity__Hz",
        "pps": "Quantity__pps",
        "mm": "Quantity__mm",
        "m": "Quantity__m",
        "mm\\*\\*2": "Quantity__mm2",
        "1/mm\\*\\*2": "Quantity__per_mm2",
        "m/s": "Quantity__m_per_s",
        "mm/s": "Quantity__mm_per_s",
    }

    # Build replacement patterns for all units
    replacements = []
    for unit, alias_name in quantity_units.items():
        pattern = base_pattern.replace("UNIT", unit)
        # Use full module path in filename: myogen.utils.types.Quantity__X.html
        replacement = f'<a class="reference internal" href="myogen.utils.types.{alias_name}.html#myogen.utils.types.{alias_name}" title="myogen.utils.types.{alias_name}"><span class="pre">{alias_name}</span></a>'
        replacements.append((pattern, replacement))

    # Add replacements for Block types (simpler pattern - just plain text in <p> tags)
    block_types = [
        "SPIKE_TRAIN__Block",
        "SURFACE_EMG__Block",
        "SURFACE_MUAP__Block",
        "INTRAMUSCULAR_EMG__Block",
        "INTRAMUSCULAR_MUAP__Block",
    ]

    for block_type in block_types:
        # Pattern: <p>BLOCK_TYPE</p> or <em>BLOCK_TYPE</em> in parameter lists
        pattern = f"<p>{block_type}</p>"
        replacement = f'<p><a class="reference internal" href="myogen.utils.types.{block_type}.html#myogen.utils.types.{block_type}" title="myogen.utils.types.{block_type}"><code class="xref py py-data docutils literal notranslate"><span class="pre">{block_type}</span></code></a></p>'
        replacements.append((pattern, replacement))

        # Also handle Block types in parameter descriptions
        pattern_em = f"<em>{block_type}</em>"
        replacement_em = f'<em><a class="reference internal" href="myogen.utils.types.{block_type}.html#myogen.utils.types.{block_type}" title="myogen.utils.types.{block_type}"><code class="xref py py-data docutils literal notranslate"><span class="pre">{block_type}</span></code></a></em>'
        replacements.append((pattern_em, replacement_em))

    # Fix truncated Block type signatures (sphinx-autodoc-typehints rendering issue)
    # Pattern: parameter name followed by truncated type hint ]]
    truncated_sig_patterns = [
        # spike_train__Block parameter with truncated ]] type
        (
            r'(<span class="n"><span class="pre">spike_train__Block</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span>)<span class="n"><span class="pre">]]</span></span>',
            r'\1<a class="reference internal" href="myogen.utils.types.SPIKE_TRAIN__Block.html#myogen.utils.types.SPIKE_TRAIN__Block" title="myogen.utils.types.SPIKE_TRAIN__Block"><span class="n"><span class="pre">SPIKE_TRAIN__Block</span></span></a>',
        ),
    ]

    replacements.extend(truncated_sig_patterns)

    # Process all HTML files with context-aware replacements
    total_replacements = 0
    for html_file in build_dir.rglob("*.html"):
        try:
            content = html_file.read_text(encoding="utf-8")
            original_content = content

            # Apply all replacement patterns
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            # Context-aware return type replacements for truncated Block types
            # SurfaceEMG methods return SURFACE_EMG__Block or SURFACE_MUAP__Block
            if "SurfaceEMG" in html_file.name:
                if "simulate_surface_emg" in content or "add_noise" in content:
                    # These return SURFACE_EMG__Block
                    content = re.sub(
                        r'(<span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint">)<span class="pre">segments\)\)]]</span>',
                        r'\1<a class="reference internal" href="myogen.utils.types.SURFACE_EMG__Block.html#myogen.utils.types.SURFACE_EMG__Block" title="myogen.utils.types.SURFACE_EMG__Block"><span class="pre">SURFACE_EMG__Block</span></a>',
                        content,
                    )
                elif "simulate_muaps" in content:
                    # This returns SURFACE_MUAP__Block
                    content = re.sub(
                        r'(<span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint">)<span class="pre">segments\)\)]]</span>',
                        r'\1<a class="reference internal" href="myogen.utils.types.SURFACE_MUAP__Block.html#myogen.utils.types.SURFACE_MUAP__Block" title="myogen.utils.types.SURFACE_MUAP__Block"><span class="pre">SURFACE_MUAP__Block</span></a>',
                        content,
                    )

            # IntramuscularEMG methods return INTRAMUSCULAR_EMG__Block or INTRAMUSCULAR_MUAP__Block
            if "IntramuscularEMG" in html_file.name:
                if "simulate_intramuscular_emg" in content or "add_noise" in content:
                    # These return INTRAMUSCULAR_EMG__Block
                    content = re.sub(
                        r'(<span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint">)<span class="pre">segments\)\)]]</span>',
                        r'\1<a class="reference internal" href="myogen.utils.types.INTRAMUSCULAR_EMG__Block.html#myogen.utils.types.INTRAMUSCULAR_EMG__Block" title="myogen.utils.types.INTRAMUSCULAR_EMG__Block"><span class="pre">INTRAMUSCULAR_EMG__Block</span></a>',
                        content,
                    )
                elif "simulate_muaps" in content:
                    # This returns INTRAMUSCULAR_MUAP__Block
                    content = re.sub(
                        r'(<span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint">)<span class="pre">segments\)\)]]</span>',
                        r'\1<a class="reference internal" href="myogen.utils.types.INTRAMUSCULAR_MUAP__Block.html#myogen.utils.types.INTRAMUSCULAR_MUAP__Block" title="myogen.utils.types.INTRAMUSCULAR_MUAP__Block"><span class="pre">INTRAMUSCULAR_MUAP__Block</span></a>',
                        content,
                    )

            # ForceModel.generate_force returns with truncated "N]]" (likely FORCE__AnalogSignal but truncated)
            if "ForceModel" in html_file.name and "generate_force" in content:
                content = re.sub(
                    r'(<span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint">)<span class="pre">N]]</span>',
                    r'\1<a class="reference internal" href="myogen.utils.types.FORCE__AnalogSignal.html#myogen.utils.types.FORCE__AnalogSignal" title="myogen.utils.types.FORCE__AnalogSignal"><span class="pre">FORCE__AnalogSignal</span></a>',
                    content,
                )

            # inject_currents_and_simulate_spike_trains returns SPIKE_TRAIN__Block
            if "inject_currents_and_simulate_spike_trains" in html_file.name:
                content = re.sub(
                    r'(<span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint">)<span class="pre">segments\)\)]]</span>',
                    r'\1<a class="reference internal" href="myogen.utils.types.SPIKE_TRAIN__Block.html#myogen.utils.types.SPIKE_TRAIN__Block" title="myogen.utils.types.SPIKE_TRAIN__Block"><span class="pre">SPIKE_TRAIN__Block</span></a>',
                    content,
                )

            # Check if anything was modified
            if content != original_content:
                html_file.write_text(content, encoding="utf-8")
                total_replacements += 1
                print(f"Processed {html_file.name}")
        except Exception as e:
            print(f"Warning: Could not process {html_file}: {e}")

    if total_replacements > 0:
        print(f"Total: Modified {total_replacements} HTML files")


def setup(app):
    """Setup function for custom configurations."""
    app.add_css_file("custom.css")
    app.add_js_file("custom.js")

    # Add custom autodoc processors
    app.connect("autodoc-process-docstring", prettify_type_alias)
    app.connect("build-finished", post_process_html)
