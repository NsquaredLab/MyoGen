# MyoGen docs migration: Sphinx → properdocs (MkDocs) + mkdocs-gallery

**Status:** Design proposal (approved). Lands on `feat/puffer-api`.
**Date:** 2026-06-23

## Goal

Replace MyoGen's Sphinx documentation stack with **properdocs** (NsquaredLab's mkdocs-material + mkdocstrings wrapper, as used by MyoGestic and Virtual-Hand-Interface), while **keeping the executed example gallery** via the **mkdocs-gallery** plugin. End state: `properdocs build` produces the complete site, examples still auto-execute and render their plots, and CI deploys to GitHub Pages.

## Why

Consistency with the other NsquaredLab projects (MyoGestic, VHI) which all use properdocs. properdocs is a thin mkdocs fork (`config.load_config` + `cfg.plugins`), on PyPI (1.6.7), and supports arbitrary mkdocs plugins — so mkdocs-gallery can be added to keep the executed gallery that sphinx-gallery currently provides.

## Non-goals

- Rewriting example *content* — the `# %%`-delimited scripts are mkdocs-gallery-compatible and stay as-is (only the gallery engine changes).
- Changing the public API or any `myogen/` code.

## Architecture

### Tooling
- New **`properdocs.yml`** at repo root (mkdocs config format), mirroring MyoGestic's: `theme: material` (custom palette, Inter/JetBrains Mono fonts, navigation features), `plugins: [search, section-index, mkdocstrings, gallery]`, and the same `markdown_extensions` (admonition, pymdownx.*, toc permalink, mermaid superfence).
- `pyproject.toml` `docs` extra: **remove** `sphinx`, `sphinx-gallery`, `pydata-sphinx-theme`, `sphinx-design`, `sphinx-hoverxref`, `sphinxcontrib-mermaid`, `sphinx-autodoc-typehints`, `enum-tools[sphinx]`, `rinohtype`, `roman`, `toml`, `linkify-it-py`, `memory-profiler`. **Add** `properdocs`, `mkdocs-material>=9.5`, `mkdocstrings[python]>=0.27`, `mkdocs-section-index>=0.3`, `mkdocs-gallery`. Keep `pynwb`/`nwbinspector` only if still referenced (NWB lives in the `nwb` extra; example execution syncs both extras in CI).

### Content layout (flat `docs/`, no `source/`)
- `docs/index.md` — home (from existing `docs/source/index.md`).
- `docs/getting-started.md` — quick start.
- `docs/neo-blocks.md` — from `docs/source/neo_blocks_guide.rst`.
- `docs/api/index.md` + `docs/api/{core,simulator,neuron,currents,utils,plotting,types}.md` — converted from the 8 rst `autosummary` pages to **mkdocstrings `:::` blocks**, preserving the section headings. mkdocstrings options mirror MyoGestic (numpy docstrings, `show_root_heading`, `merge_init_into_class`, `filters: ["!^_"]`).
- `nav:` in `properdocs.yml` lists Home / Getting Started / Neo blocks / Examples (gallery) / API reference.
- **`exclude_docs:`** in the config globs out `superpowers/` so the brainstorming specs/plans under `docs/superpowers/` are not built as site pages.
- Static assets (logo, any custom CSS) migrated from `docs/source/_static/` to `docs/` (e.g. `docs/stylesheets/`, `docs/images/`).

### Executed gallery (mkdocs-gallery)
- The `gallery` plugin block in `properdocs.yml` ports the current `sphinx_gallery_conf` ~1:1:
  - `examples_dirs: [examples/01_basic, examples/02_finetune, examples/03_papers/watanabe]`
  - `gallery_dirs: [docs/auto_examples/01_basic, docs/auto_examples/02_finetune, docs/auto_examples/03_papers/watanabe]` (generated; gitignored)
  - `filename_pattern: "\\.py"`, `ignore_pattern: "(14_calibrate_noise_from_real|_oscillating_dc_helpers|_optimize_dc_worker)\\.py"`
  - `within_subsection_order: FileNameSortKey`, explicit subsection order, `plot_gallery: True`, `image_scrapers: matplotlib`.
  - `reset_modules`: the existing `reset_neuron(gallery_conf, fname)` (resets NEURON `h` state between examples), moved to **`docs/gallery_conf.py`** and referenced from the plugin config.
- The generated `auto_examples/` tree is added to `.gitignore` and to the nav (or surfaced via section-index).

### CI
- Delete `.github/workflows/sphinx.yml`; add `.github/workflows/docs.yml` mirroring MyoGestic's: build job runs `uv sync --locked --extra docs --extra nwb` (so NEURON + example deps are present and the gallery can execute), then `uv run --locked --extra docs --extra nwb properdocs build`, uploads the `site/` Pages artifact; deploy job publishes to GitHub Pages on push to `main` (PRs build-only). Concurrency group `pages`.

### Removal
Delete `docs/source/conf.py`, `docs/source/examples.rst`, `docs/source/api/*.rst`, `docs/source/index.rst.backup`, `docs/source/neo_blocks_guide.rst` (after conversion), `docs/source/templates/`, `docs/Makefile`, `docs/make.bat`, and the Sphinx `_static`/`source/` scaffolding once content is migrated.

## Verification

1. `uv sync --extra docs --extra nwb` resolves.
2. `properdocs build` with the gallery temporarily set to `plot_gallery: False` → confirms config/theme/nav/mkdocstrings all resolve fast (no example execution). Open `site/index.html`.
3. Full `properdocs build` (gallery executes; watanabe/02 now ~139s) → confirms examples render with plots. Open the built gallery pages.
4. mkdocstrings resolves every API symbol (requires a full `import myogen`, which needs NEURON — present via the synced extras).
5. Show the rendered site to the user at each checkpoint.

## Risks / mitigations

- **mkdocs-gallery ↔ properdocs**: properdocs is mkdocs-based and infers deps from the `plugins:` list, so the `gallery` plugin should load; verified at build time in step 2/3.
- **CI build time**: executing the gallery is the same cost as today's sphinx CI; the slowest example (watanabe/02) is now 5× faster. The `14`/helper/worker ignores keep non-executable scripts out.
- **`docs/superpowers/` leaking into the site**: handled by `exclude_docs`.
- **mkdocstrings import failures**: any optional dep referenced by an API symbol must be installed in the build env — covered by syncing `--extra docs --extra nwb`.
