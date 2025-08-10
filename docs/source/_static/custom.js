document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('dd').forEach(dd => {
    // Check if last child is a text node containing comma
    const lastChild = dd.lastChild;
    if (lastChild && lastChild.nodeType === Node.TEXT_NODE) {
      // Remove the comma (and trim surrounding spaces)
      lastChild.textContent = lastChild.textContent.replace(/,/, '');
    }
  });
});
