document.addEventListener('DOMContentLoaded', () => {
  // Remove trailing commas from dd elements
  document.querySelectorAll('dd').forEach(dd => {
    // Check if last child is a text node containing comma
    const lastChild = dd.lastChild;
    if (lastChild && lastChild.nodeType === Node.TEXT_NODE) {
      // Remove the comma (and trim surrounding spaces)
      lastChild.textContent = lastChild.textContent.replace(/,/, '');
    }
  });

  // Format type alias annotations with proper indentation
  document.querySelectorAll('dl.py.data dd p').forEach(elem => {
    const text = elem.textContent;

    // Only process if it contains "alias of Annotated"
    if (text.includes('alias of') && text.includes('Annotated')) {
      // Clone the element to work with
      const clone = elem.cloneNode(true);

      // Get the HTML content
      const html = clone.innerHTML;

      // Check if it starts with "alias of"
      if (html.includes('alias of')) {
        // Format the HTML while preserving links
        const formatted = formatTypeAnnotationHTML(clone);

        // Replace content
        elem.innerHTML = '<strong>alias of</strong>' + formatted;
      }
    }
  });
});

function formatTypeAnnotationHTML(elem) {
  // Remove "alias of" text if present
  const walker = document.createTreeWalker(
    elem,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  let node;
  while (node = walker.nextNode()) {
    node.textContent = node.textContent.replace('alias of', '').trim();
  }

  // Get the HTML content after removing "alias of"
  let html = elem.innerHTML.trim();

  // Format the HTML with proper indentation
  let result = '\n<pre><code>';
  let indentLevel = 0;
  const indentSize = 4;
  let insideTag = false;
  let currentTag = '';
  let i = 0;

  while (i < html.length) {
    const char = html[i];

    if (char === '<') {
      insideTag = true;
      currentTag = '';
      result += char;
    } else if (char === '>') {
      insideTag = false;
      result += char;
      currentTag = '';
    } else if (insideTag) {
      currentTag += char;
      result += char;
    } else if (char === '[') {
      result += char + '\n';
      indentLevel++;
      result += ' '.repeat(indentLevel * indentSize);
    } else if (char === ']') {
      result += '\n';
      indentLevel--;
      result += ' '.repeat(indentLevel * indentSize) + char;
    } else if (char === ',' && !insideTag) {
      result += char + '\n';
      // Skip following space
      if (i + 1 < html.length && html[i + 1] === ' ') {
        i++;
      }
      result += ' '.repeat(indentLevel * indentSize);
    } else {
      result += char;
    }

    i++;
  }

  result += '</code></pre>';

  // Add links for common types
  result = addTypeLinks(result);

  // Wrap in proper Sphinx highlight div for consistent styling
  return '<div class="highlight-python notranslate"><div class="highlight">' + result + '</div></div>';
}

function addTypeLinks(html) {
  // Define link mappings with Pygments classes for syntax highlighting
  const links = {
    'Annotated': {
      url: 'https://docs.python.org/3/library/typing.html#typing.Annotated',
      class: 'nc' // class name
    },
    'Quantity': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html#quantity-basics',
      class: 'nc' // class name
    },
    'beartype.vale.IsAttr': {
      url: 'https://beartype.readthedocs.io/en/latest/api_vale/#beartype.vale.IsAttr',
      class: 'n' // name
    },
    'beartype.vale.Is': {
      url: 'https://beartype.readthedocs.io/en/latest/api_vale/#beartype.vale.Is',
      class: 'n' // name
    },
    'beartype.vale.IsEqual': {
      url: 'https://beartype.readthedocs.io/en/latest/api_vale/#beartype.vale.IsEqual',
      class: 'n' // name
    },
    'beartype': {
      url: 'https://beartype.readthedocs.io/',
      class: 'nn' // module name
    },
    // Quantities units
    'pq.nA': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.s': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.ms': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.mV': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.uV': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.Hz': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.mm': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.uS': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.rad': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.deg': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.m': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.S': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
    'pq.N': {
      url: 'https://python-quantities.readthedocs.io/en/latest/user/tutorial.html',
      class: 'n' // name
    },
  };

  // Sort by length (longest first) to avoid replacing parts of longer names
  const sortedKeys = Object.keys(links).sort((a, b) => b.length - a.length);

  for (const term of sortedKeys) {
    const {url, class: pygmentsClass} = links[term];
    // Escape dots in term for regex
    const escapedTerm = term.replace(/\./g, '\\.');

    // Split by < to work with text nodes only, avoiding already-linked content
    const parts = html.split(/(<[^>]+>)/);
    for (let i = 0; i < parts.length; i++) {
      // Only process text parts (not HTML tags)
      if (!parts[i].startsWith('<')) {
        const regex = new RegExp(`\\b(${escapedTerm})\\b`, 'g');
        parts[i] = parts[i].replace(regex,
          `<a href="${url}" class="reference external ${pygmentsClass}" title="(external link)"><span class="${pygmentsClass}">$1</span></a>`
        );
      }
    }
    html = parts.join('');
  }

  // Add syntax highlighting for strings (anything in quotes)
  html = html.replace(/'([^']*)'/g, '<span class="s1">\'$1\'</span>');

  // Add syntax highlighting for operators
  html = html.replace(/(\[|\]|,)/g, '<span class="p">$1</span>');

  return html;
}
