/**
 * Safe Markdown Parser & Sanitizer
 * Formats code blocks, bold/italics, lists, and line breaks while preventing XSS.
 */

const SafeMarkdown = {
  escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  },

  render(markdownText) {
    if (!markdownText) return "";

    let text = this.escapeHtml(markdownText);

    // Code blocks with copy button
    text = text.replace(/```([a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || "text";
      return `
        <div class="code-block-wrapper">
          <div class="code-header">
            <span>${language}</span>
            <button class="btn-action" onclick="navigator.clipboard.writeText(this.closest('.code-block-wrapper').querySelector('code').innerText); this.innerText='Copied!'; setTimeout(()=>this.innerText='Copy', 1500);">
              Copy
            </button>
          </div>
          <pre class="code-block"><code>${code.trim()}</code></pre>
        </div>
      `;
    });

    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code class="metrics-badge">$1</code>');

    // Bold & Italics
    text = text.replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>");
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Bulleted lists
    text = text.replace(/^\s*-\s+(.+)$/gm, '<li style="margin-left: 20px;">$1</li>');

    // Paragraph line breaks
    text = text.replace(/\n\n+/g, "</p><p>");
    text = text.replace(/\n/g, "<br/>");

    return `<p>${text}</p>`;
  }
};
