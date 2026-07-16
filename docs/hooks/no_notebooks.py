"""mkdocs hook: never ship Jupyter notebooks from the example gallery.

mkdocs-gallery hard-codes a "Download Jupyter notebook" button into every
rendered example and writes a sibling ``.ipynb`` next to each example's source.
There is no upstream toggle to disable this, so we enforce it here in two
complementary ways:

1. ``on_page_content`` strips the notebook download button from the rendered
   HTML, so it never appears in the UI.
2. ``on_post_build`` deletes every ``.ipynb`` from the built site, so the files
   cannot be reached by direct URL either.

Registered via ``hooks:`` in ``properdocs.yml``.
"""

import os
import re

# The notebook button is a markdown link rendered to a <p> wrapping an <a> whose
# href ends in .ipynb. Match that paragraph (and only that one) for removal.
_NOTEBOOK_BUTTON = re.compile(
    r"<p>\s*<a[^>]*href=\"[^\"]*\.ipynb\"[^>]*>.*?</a>\s*</p>",
    re.DOTALL | re.IGNORECASE,
)


def on_page_content(html, page, config, files):
    """Remove the 'Download Jupyter notebook' button from gallery pages."""
    if ".ipynb" in html:
        return _NOTEBOOK_BUTTON.sub("", html)
    return html


def on_post_build(config):
    """Delete every generated .ipynb from the built site."""
    site_dir = config["site_dir"]
    removed = 0
    for root, _dirs, names in os.walk(site_dir):
        for name in names:
            if name.endswith(".ipynb"):
                os.remove(os.path.join(root, name))
                removed += 1
    if removed:
        from mkdocs import utils

        utils.log.info("no_notebooks hook: removed %d .ipynb file(s) from site", removed)
