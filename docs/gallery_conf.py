"""mkdocs-gallery base configuration for the MyoGen example gallery.

Referenced from ``properdocs.yml`` via the ``gallery`` plugin's ``conf_script``.
The module-level ``conf`` dict is loaded as the gallery's base configuration;
``plot_gallery`` is overridden from ``properdocs.yml`` (env-controlled).
"""

from pathlib import Path

import matplotlib
from mkdocs_gallery.sorting import FileNameSortKey

# Headless build: render figures off-screen and capture them, never try to
# display. mkdocs-gallery's matplotlib scraper collects figures via
# ``plt.get_fignums()`` (see mkdocs_gallery/scrapers.py), independently of
# ``plt.show()``. Under the Agg backend ``plt.show()`` therefore does nothing
# useful and only emits a "FigureCanvasAgg is non-interactive, and thus cannot
# be shown" UserWarning that leaks (with the absolute source path) into every
# example's Out block. sphinx-gallery no-ops ``show`` for exactly this reason;
# mkdocs-gallery does not, so we do it here.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

plt.show = lambda *args, **kwargs: None

# mkdocs-gallery requires examples_dirs / gallery_dirs as absolute paths under
# the project root (it calls Path(...).relative_to(project_root)). This file
# lives at docs/gallery_conf.py, so its grandparent is the repo root.
_ROOT = Path(__file__).resolve().parent.parent


def reset_neuron(gallery_conf, fname):
    """Reset NEURON global HOC state between gallery examples.

    NEURON's interpreter keeps process-global state across examples run in one
    process; clearing sections + time between examples keeps them independent.
    """
    try:
        import myogen  # noqa: F401  (auto-loads mechanisms, sets up NEURON)
        from neuron import h

        for sec in list(h.allsec()):
            h.delete_section(sec=sec)
        h.load_file("stdrun.hoc")
        h.t = 0
        h.tstop = 0
    except (ImportError, RuntimeError, LookupError, AttributeError):
        pass


conf = {
    "examples_dirs": [
        str(_ROOT / "examples" / "01_basic"),
        str(_ROOT / "examples" / "02_finetune"),
        str(_ROOT / "examples" / "03_papers" / "watanabe"),
    ],
    "gallery_dirs": [
        str(_ROOT / "docs" / "auto_examples" / "01_basic"),
        str(_ROOT / "docs" / "auto_examples" / "02_finetune"),
        str(_ROOT / "docs" / "auto_examples" / "03_papers" / "watanabe"),
    ],
    "filename_pattern": r"\.py",
    "ignore_pattern": r"(14_calibrate_noise_from_real|_oscillating_dc_helpers|_optimize_dc_worker)\.py",
    "within_subsection_order": FileNameSortKey,
    "image_scrapers": ("matplotlib",),
    # strip the `# mkdocs_gallery_thumbnail_path = ...` directives from rendered source
    "remove_config_comments": True,
    # fallback thumbnail for examples without a captured figure (logo)
    "default_thumb_file": str(_ROOT / "docs" / "images" / "myogen_logo.png"),
    "reset_modules": (reset_neuron,),
    # Source-only by default; overridden from properdocs.yml via MKDOCS_GALLERY_PLOT.
    "plot_gallery": False,
}
