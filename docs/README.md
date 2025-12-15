# MyoGen Documentation

This directory contains the source files for MyoGen's Sphinx documentation.

## Building the Documentation

```bash
# From the docs directory
make html
```

The built documentation will be in `build/html/`.

## Viewing the Documentation

**Important:** Hover tooltips (hoverxref) require viewing the documentation via HTTP server, not by opening `file://` URLs directly.

### Option 1: Using the serve script (recommended)

```bash
cd docs
./serve_docs.sh
```

Then open http://localhost:8000 in your browser.

### Option 2: Manual Python server

```bash
cd docs/build/html
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Why is this necessary?

The hoverxref extension uses AJAX to fetch tooltip content. Browsers block AJAX requests from `file://` URLs for security reasons, causing tooltips to show "Loading..." indefinitely. Serving via HTTP resolves this issue.

## Documentation Structure

- `source/` - RST source files
  - `api/` - API reference documentation
  - `auto_examples/` - Generated example galleries (auto-generated)
  - `_static/` - Static files (CSS, JS, images)
  - `templates/` - Custom Sphinx templates
  - `generated/` - Auto-generated API docs (auto-generated)
- `build/` - Built documentation (generated, not tracked in git)

## Making Changes

1. Edit RST files in `source/`
2. Rebuild: `make html`
3. View changes: `./serve_docs.sh`

## Cleaning Build

```bash
make clean
```
