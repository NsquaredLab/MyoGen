# MyoGen Docs → properdocs (MkDocs) Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MyoGen's Sphinx docs with properdocs (mkdocs-material + mkdocstrings) while keeping the executed example gallery via mkdocs-gallery; CI deploys to GitHub Pages.

**Architecture:** A repo-root `properdocs.yml` (mkdocs config) drives a `material` theme, `mkdocstrings` API pages, and a `gallery` plugin that executes the `examples/` scripts. Content is flat markdown under `docs/`; `docs/superpowers/` is excluded from the build. A `docs.yml` GitHub workflow runs `properdocs build` (executing the gallery) and deploys.

**Tech Stack:** properdocs 1.6.x, mkdocs-material, mkdocstrings[python], mkdocs-section-index, mkdocs-gallery; uv for env management.

**Reference:** mirror `/Users/oj98yqyk/code/MyoGestic-main/properdocs.yml` and `.github/workflows/docs.yml`. Source spec: `docs/superpowers/specs/2026-06-23-mkdocs-properdocs-migration-design.md`.

**Verification model:** there are no pytest tests here — each task's "test" is a build/inspection. The canonical fast build (no example execution) is:

```bash
cd /Users/oj98yqyk/code/MyoGen
MKDOCS_GALLERY_PLOT=false uv run --extra docs --extra nwb properdocs build 2>&1 | tail -30
```

(Until the gallery plugin exists, drop the env var.) "Open the site" means `open site/index.html`.

---

### Task 1: Swap the `docs` extra in pyproject.toml

**Files:** Modify `pyproject.toml` (the `docs = [...]` block, ~lines 82-100); Modify `uv.lock`.

- [ ] **Step 1: Replace the docs extra**

In `pyproject.toml`, replace the entire `docs = [ ... ]` list under `[project.optional-dependencies]` with:

```toml
docs = [
    "properdocs>=1.6.7",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.27",
    "mkdocs-section-index>=0.3",
    "mkdocs-gallery>=0.10",
]
```

(The `nwb` extra already carries `pynwb`/`nwbinspector`/`h5py`; example execution syncs `--extra docs --extra nwb` together. Sphinx deps are removed.)

- [ ] **Step 2: Lock and verify resolution**

Run:
```bash
uv lock 2>&1 | tail -3
uv sync --extra docs --extra nwb 2>&1 | tail -5
```
Expected: `uv lock` resolves; `uv sync` installs properdocs + mkdocs-material + mkdocstrings + mkdocs-section-index + mkdocs-gallery with no error. Confirm the binary exists:
```bash
uv run properdocs --version
```
Expected: prints a version (e.g. `properdocs, version 1.6.7`).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(docs): swap Sphinx stack for properdocs + mkdocs-gallery"
```

---

### Task 2: properdocs.yml skeleton + home page (theme/nav, no gallery yet)

**Files:** Create `properdocs.yml`; Create `docs/index.md` (temporary minimal home, replaced in Task 4); Modify `.gitignore`.

- [ ] **Step 1: Create `properdocs.yml`**

Create `properdocs.yml` at the repo root (adapted from MyoGestic's; gallery plugin added in Task 5):

```yaml
site_name: MyoGen
site_description: Modular neuromuscular simulation framework for motor-unit activity, force, and EMG.
site_url: https://nsquaredlab.github.io/MyoGen/
repo_url: https://github.com/NsquaredLab/MyoGen
repo_name: MyoGen
edit_uri: edit/main/docs/
copyright: Copyright &copy; 2025-2026 n-squared lab, FAU Erlangen-Nürnberg

# superpowers specs/plans live under docs/ but are not site pages
exclude_docs: |
  superpowers/

theme:
  name: material
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.code.annotate
    - toc.follow
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: white
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: black
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  font:
    text: Inter
    code: JetBrains Mono
  icon:
    repo: fontawesome/brands/github

plugins:
  - search
  - section-index
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          inventories:
            - https://docs.python.org/3/objects.inv
            - https://numpy.org/doc/stable/objects.inv
            - https://docs.scipy.org/doc/scipy/objects.inv
          options:
            docstring_style: numpy
            show_source: true
            show_root_heading: true
            show_root_full_path: false
            members_order: source
            separate_signature: true
            show_signature_annotations: true
            signature_crossrefs: true
            merge_init_into_class: true
            heading_level: 2
            filters:
              - "!^_"

markdown_extensions:
  - admonition
  - attr_list
  - def_list
  - md_in_html
  - tables
  - toc:
      permalink: true
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.highlight:
      anchor_linenums: true
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.magiclink
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg

nav:
  - Home: index.md
```

- [ ] **Step 2: Create a minimal home page**

Create `docs/index.md`:

```markdown
# MyoGen

Modular, extensible neuromuscular simulation framework for generating
physiologically grounded motor-unit activity, muscle force, and EMG
(surface and intramuscular).

This site is built with [properdocs](https://pypi.org/project/properdocs/).
```

- [ ] **Step 3: Ignore build outputs**

Append to `.gitignore`:

```gitignore
# mkdocs / properdocs build output
/site/
# mkdocs-gallery generated pages
/docs/auto_examples/
```

- [ ] **Step 4: Build and verify**

Run:
```bash
uv run --extra docs --extra nwb properdocs build 2>&1 | tail -20
```
Expected: build succeeds, writes `site/`. Then:
```bash
test -f site/index.html && echo "OK site built" && open site/index.html
```
Expected: `OK site built`; the page opens showing the MyoGen home with the material theme (light/dark toggle, MyoGen title).

- [ ] **Step 5: Commit**

```bash
git add properdocs.yml docs/index.md .gitignore
git commit -m "docs: add properdocs.yml skeleton with material theme + home page"
```

---

### Task 3: API reference pages via mkdocstrings

**Files:** Create `docs/api/index.md`, `docs/api/simulator.md`, `docs/api/neuron.md`, `docs/api/currents.md`, `docs/api/io-and-neuron.md`, `docs/api/plotting.md`, `docs/api/types.md`, `docs/api/myogen.md`; Modify `properdocs.yml` (nav).

The conversion rule: each old rst `autosummary` entry under a `.. currentmodule:: X` becomes a mkdocstrings block `::: X.Symbol`. Keep the section headings as markdown `##`/`###`.

- [ ] **Step 1: Create `docs/api/index.md`**

```markdown
# API Reference

- [Top-level (`myogen`)](myogen.md)
- [Simulator](simulator.md)
- [Currents](currents.md)
- [Neuron injection & I/O](io-and-neuron.md)
- [Plotting](plotting.md)
- [Types](types.md)
```

- [ ] **Step 2: Create `docs/api/myogen.md`** (from `myogen_api.rst`)

```markdown
# Top-level API (`myogen`)

::: myogen.set_random_seed
::: myogen.get_random_generator
::: myogen.get_random_seed
::: myogen.derive_subseed
::: myogen.load_nmodl_mechanisms
::: myogen.get_mechanism_parameters
::: myogen.validate_mechanism_parameter
::: myogen.set_mechanism_param
```

- [ ] **Step 3: Create `docs/api/simulator.md`** (from `simulator_api.rst`)

```markdown
# Simulator

## Recruitment
::: myogen.simulator.RecruitmentThresholds

## Neuron populations
::: myogen.simulator.neuron.populations.AlphaMN__Pool
::: myogen.simulator.neuron.populations.DescendingDrive__Pool
::: myogen.simulator.neuron.populations.AffIa__Pool
::: myogen.simulator.neuron.populations.AffII__Pool
::: myogen.simulator.neuron.populations.AffIb__Pool
::: myogen.simulator.neuron.populations.GII__Pool
::: myogen.simulator.neuron.populations.GIb__Pool

## Network & runner
::: myogen.simulator.neuron.network.Network
::: myogen.simulator.neuron.simulation_runner.SimulationRunner

## Muscle & force
::: myogen.simulator.Muscle
::: myogen.simulator.neuron.muscle.HillModel
::: myogen.simulator.ForceModel
::: myogen.simulator.ForceModelVectorized

## EMG
::: myogen.simulator.SurfaceEMG
::: myogen.simulator.IntramuscularEMG
::: myogen.simulator.SurfaceElectrodeArray
::: myogen.simulator.IntramuscularElectrodeArray

## Proprioception
::: myogen.simulator.neuron.proprioception.SpindleModel
::: myogen.simulator.neuron.proprioception.GolgiTendonOrganModel
::: myogen.simulator.neuron.joint_dynamics.JointDynamics
```

- [ ] **Step 4: Create `docs/api/currents.md`** (from `currents_api.rst`)

```markdown
# Currents

::: myogen.utils.currents.create_ramp_current
::: myogen.utils.currents.create_step_current
::: myogen.utils.currents.create_sinusoidal_current
::: myogen.utils.currents.create_sawtooth_current
::: myogen.utils.currents.create_trapezoid_current
```

- [ ] **Step 5: Create `docs/api/io-and-neuron.md`** (from `utils_api.rst` injection + saver + nwb)

```markdown
# Neuron injection & I/O

## Current injection
::: myogen.utils.neuron.inject_currents_into_populations.inject_currents_into_populations
::: myogen.utils.neuron.inject_currents_into_populations.inject_currents_and_simulate_spike_trains

## Persistence
::: myogen.utils.continuous_saver.ContinuousSaver
::: myogen.utils.continuous_saver.convert_chunks_to_neo

## NWB export
!!! note
    NWB export requires optional dependencies: `pip install myogen[nwb]`.

::: myogen.utils.nwb.export_to_nwb
::: myogen.utils.nwb.export_simulation_to_nwb
::: myogen.utils.nwb.validate_nwb
```

- [ ] **Step 6: Create `docs/api/plotting.md`** (from `plotting_api.rst`)

```markdown
# Plotting

::: myogen.utils.plotting.plot_raster_spikes
::: myogen.utils.plotting.plot_membrane_potentials
::: myogen.utils.plotting.plot_muscle_dynamics
::: myogen.utils.plotting.plot_antagonist_muscle_comparison
::: myogen.utils.plotting.plot_spindle_dynamics
::: myogen.utils.plotting.plot_gto_dynamics
```

- [ ] **Step 7: Create `docs/api/types.md`** (from `types_api.rst`)

```markdown
# Types

::: myogen.utils.types
    options:
      members:
        - Quantity__s
        - Quantity__ms
        - Quantity__rad
        - Quantity__deg
        - Quantity__mV
        - Quantity__uV
        - Quantity__nA
        - Quantity__uS
        - Quantity__S_per_m
        - Quantity__Hz
        - Quantity__pps
        - Quantity__mm
        - Quantity__m
        - Quantity__mm2
        - Quantity__per_mm2
        - Quantity__m_per_s
        - Quantity__mm_per_s
        - CURRENT__AnalogSignal
        - FORCE__AnalogSignal
        - SPIKE_TRAIN__Block
        - SURFACE_MUAP__Block
        - SURFACE_EMG__Block
        - INTRAMUSCULAR_MUAP__Block
        - INTRAMUSCULAR_EMG__Block
```

- [ ] **Step 8: Add API to nav**

In `properdocs.yml`, extend `nav:` to:

```yaml
nav:
  - Home: index.md
  - API reference:
      - api/index.md
      - Top-level: api/myogen.md
      - Simulator: api/simulator.md
      - Currents: api/currents.md
      - Injection & I/O: api/io-and-neuron.md
      - Plotting: api/plotting.md
      - Types: api/types.md
```

- [ ] **Step 9: Build and verify mkdocstrings resolves every symbol**

Run:
```bash
uv run --extra docs --extra nwb properdocs build --strict 2>&1 | tail -40
```
Expected: build succeeds with NO "Could not collect" / griffe import warnings. If any symbol path is wrong (e.g. a class lives in a different module), fix the `:::` path against the real location in `myogen/` and rebuild. Then:
```bash
open site/api/simulator/index.html
```
Expected: rendered API docs with signatures + numpy docstrings for `AlphaMN__Pool`, `SurfaceEMG`, etc.

- [ ] **Step 10: Commit**

```bash
git add docs/api properdocs.yml
git commit -m "docs: add mkdocstrings API reference pages"
```

---

### Task 4: Prose pages (home, getting-started, neo blocks) + assets

**Files:** Rewrite `docs/index.md`; Create `docs/getting-started.md`, `docs/neo-blocks.md`; Create `docs/stylesheets/` and `docs/images/` as needed; Modify `properdocs.yml` (nav, logo, extra_css).

- [ ] **Step 1: Real home page**

Replace `docs/index.md` with a landing page adapted from the existing `docs/source/index.md` and the project README: a one-paragraph intro, an install snippet (`uv add myogen` / `pip install myogen`), and cards/links to Getting Started, the Examples gallery, and the API reference. Use the existing `docs/source/index.md` content as the source of truth — read it and port the prose to markdown (drop Sphinx directives; convert `{toctree}`/`grid` to a plain markdown list or `attr_list` cards).

- [ ] **Step 2: Getting Started**

Create `docs/getting-started.md` from the install + quick-start prose in the project `README.md` (root). Port the minimal end-to-end snippet (create pool → inject current → simulate → force/EMG) as a fenced `python` block.

- [ ] **Step 3: Neo blocks guide**

Create `docs/neo-blocks.md` by converting `docs/source/neo_blocks_guide.rst` to markdown (headings, code-blocks, lists). Read the rst and translate directives: `.. code-block:: python` → ```` ```python ````, `.. note::` → `!!! note`, cross-refs → plain text or markdown links.

- [ ] **Step 4: Assets**

If `docs/source/_static/` contains a logo or custom CSS still wanted, copy the logo to `docs/images/` and any CSS to `docs/stylesheets/`, and reference them in `properdocs.yml` (`theme.logo`, `theme.favicon`, `extra_css`). If there are no assets worth keeping, skip and note it in the commit.

- [ ] **Step 5: Nav**

Update `properdocs.yml` `nav:` to:

```yaml
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Neo blocks: neo-blocks.md
  - API reference:
      - api/index.md
      - Top-level: api/myogen.md
      - Simulator: api/simulator.md
      - Currents: api/currents.md
      - Injection & I/O: api/io-and-neuron.md
      - Plotting: api/plotting.md
      - Types: api/types.md
```

- [ ] **Step 6: Build and verify**

Run:
```bash
uv run --extra docs --extra nwb properdocs build --strict 2>&1 | tail -20
open site/index.html
```
Expected: build clean; home, getting-started, and neo-blocks render with working nav.

- [ ] **Step 7: Commit**

```bash
git add docs/index.md docs/getting-started.md docs/neo-blocks.md docs/images docs/stylesheets properdocs.yml
git commit -m "docs: migrate home, getting-started, and neo-blocks prose to markdown"
```

---

### Task 5: mkdocs-gallery — executed example gallery

**Files:** Create `docs/gallery_conf.py`; Modify `properdocs.yml` (add `gallery` plugin + nav entry).

- [ ] **Step 1: Gallery reset hook**

Create `docs/gallery_conf.py` (ported from the old `conf.py` `reset_neuron`):

```python
"""mkdocs-gallery configuration helpers for MyoGen examples."""


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
```

- [ ] **Step 2: Add the gallery plugin to `properdocs.yml`**

Insert into the `plugins:` list (after `mkdocstrings`):

```yaml
  - gallery:
      examples_dirs:
        - examples/01_basic
        - examples/02_finetune
        - examples/03_papers/watanabe
      gallery_dirs:
        - docs/auto_examples/01_basic
        - docs/auto_examples/02_finetune
        - docs/auto_examples/03_papers/watanabe
      filename_pattern: "\\.py"
      ignore_pattern: "(14_calibrate_noise_from_real|_oscillating_dc_helpers|_optimize_dc_worker)\\.py"
      within_subsection_order: mkdocs_gallery.sorting.FileNameSortKey
      image_scrapers: matplotlib
      reset_modules:
        - !!python/name:gallery_conf.reset_neuron
      plot_gallery: !ENV [MKDOCS_GALLERY_PLOT, true]
```

Note: `paths: [.]` already on mkdocstrings puts repo root on `sys.path`; mkdocs-gallery resolves `!!python/name:gallery_conf.reset_neuron` because `docs/` is added to the path by the gallery plugin. If the name fails to resolve at build time, change the reference to `docs.gallery_conf.reset_neuron` and add an empty `docs/__init__.py`, OR move `gallery_conf.py` to the repo root and keep `gallery_conf.reset_neuron`. Verify by build (next step). The `!ENV` makes `MKDOCS_GALLERY_PLOT=false` skip execution for fast iteration.

- [ ] **Step 3: Add the gallery to nav**

Add under `nav:` (after Neo blocks):

```yaml
  - Examples:
      - Basics: auto_examples/01_basic/index.md
      - Fine-tuning: auto_examples/02_finetune/index.md
      - Watanabe (paper): auto_examples/03_papers/watanabe/index.md
```

- [ ] **Step 4: Fast build (no execution) to validate gallery wiring**

Run:
```bash
MKDOCS_GALLERY_PLOT=false uv run --extra docs --extra nwb properdocs build 2>&1 | tail -40
```
Expected: build succeeds; `docs/auto_examples/01_basic/index.md` and per-example pages are generated from the `# %%` blocks (source shown, no executed output). If `!!python/name:gallery_conf...` fails to resolve, apply the fallback in Step 2. Then:
```bash
open site/auto_examples/01_basic/index.html
```
Expected: a gallery index listing the basic examples with their headers.

- [ ] **Step 5: Full build (executes examples) — the real verification**

Run (this executes all gallery examples; watanabe/02 is parallelized so ~minutes, not tens of minutes):
```bash
uv run --extra docs --extra nwb properdocs build 2>&1 | tail -40
```
Expected: build succeeds; example pages now contain rendered plots. Spot-check:
```bash
open site/auto_examples/01_basic/index.html
```
Expected: thumbnails/plots rendered for the basic examples. If a specific example errors during execution, fix the example or extend `ignore_pattern` (and note why), then rebuild.

- [ ] **Step 6: Commit**

```bash
git add docs/gallery_conf.py properdocs.yml
git commit -m "docs: add executed example gallery via mkdocs-gallery"
```

---

### Task 6: GitHub Pages CI workflow

**Files:** Create `.github/workflows/docs.yml`; Delete `.github/workflows/sphinx.yml`.

- [ ] **Step 1: Create `.github/workflows/docs.yml`** (mirroring MyoGestic's)

```yaml
name: Docs

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - name: Set up uv
        uses: astral-sh/setup-uv@v8.1.0
        with:
          enable-cache: true
      - name: Sync project + docs extras
        run: uv sync --locked --extra docs --extra nwb
      - name: Build docs site (executes the example gallery)
        run: uv run --locked --extra docs --extra nwb properdocs build
      - name: Upload Pages artifact
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: actions/upload-pages-artifact@v5
        with:
          path: site

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deploy.outputs.page_url }}
    steps:
      - name: Configure Pages
        uses: actions/configure-pages@v6
      - name: Deploy to GitHub Pages
        id: deploy
        uses: actions/deploy-pages@v5
```

- [ ] **Step 2: Remove the Sphinx workflow**

```bash
git rm .github/workflows/sphinx.yml
```

- [ ] **Step 3: Validate the workflow YAML**

Run:
```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml')); print('docs.yml valid YAML')"
```
Expected: `docs.yml valid YAML`. (Note: the NEURON-in-CI build path mirrors the old sphinx.yml, which already executed the gallery; if `sphinx.yml` installed extra system deps for NEURON, port those `apt`/setup steps into `docs.yml`'s build job — read `sphinx.yml` before deleting and carry over any NEURON system setup.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci(docs): build + deploy properdocs site to GitHub Pages; drop sphinx workflow"
```

---

### Task 7: Remove the Sphinx scaffolding + final build

**Files:** Delete `docs/source/` (conf.py, examples.rst, api/*.rst, index.md, neo_blocks_guide.rst, index.rst.backup, templates/, _static/ leftovers, README.md), `docs/Makefile`, `docs/make.bat`, `docs/README.md`/`docs/neo_blocks.md` if superseded.

- [ ] **Step 1: Confirm nothing still references `docs/source/`**

Run:
```bash
grep -rIl "docs/source\|sphinx_gallery\|conf.py" --include="*.py" --include="*.toml" --include="*.yml" --include="*.cfg" . | grep -v site/ | grep -v .venv/
```
Expected: no references outside the files about to be deleted. If `pyproject.toml`/`README.md` link to the old docs paths, update them to the new site URL/paths.

- [ ] **Step 2: Delete the Sphinx tree**

```bash
git rm -r docs/source docs/Makefile docs/make.bat
git rm docs/neo_blocks.md docs/README.md 2>/dev/null || true   # only if superseded by docs/neo-blocks.md / docs/index.md
```
(Keep `docs/superpowers/` — it's excluded from the build, not part of the site.)

- [ ] **Step 3: Final clean full build**

```bash
uv run --extra docs --extra nwb properdocs build --strict 2>&1 | tail -30
test -f site/index.html && echo "FINAL SITE OK" && open site/index.html
```
Expected: `--strict` build succeeds end-to-end (home, getting-started, neo-blocks, all API pages, executed gallery), no broken nav/links, `FINAL SITE OK`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: remove Sphinx scaffolding after properdocs migration"
```

---

## Final verification

```bash
uv run --extra docs --extra nwb properdocs build --strict
open site/index.html
```
The site builds strictly, the gallery executes and renders plots, and `docs/superpowers/` is absent from the site. The branch is ready to push (CI will reproduce the build + deploy on merge to `main`).
