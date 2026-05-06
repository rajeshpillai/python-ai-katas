import { createEffect, createMemo } from "solid-js";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import "./markdown-content.css";
import "./markdown-highlight.css";

hljs.registerLanguage("python", python);
hljs.registerLanguage("rust", rust);

interface MarkdownContentProps {
  source: string;
}

function parseMarkdown(md: string): string {
  let html = md;

  // Step 1: Extract code blocks into placeholders so later regexes
  // (headers, lists, etc.) cannot corrupt content inside them.
  const codeBlocks: string[] = [];
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, lang, code) => {
    const escaped = code
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    const block = `<pre class="md-code-block"><code class="language-${lang}">${escaped}</code></pre>`;
    codeBlocks.push(block);
    return `\n__CODE_BLOCK_${codeBlocks.length - 1}__\n`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="md-inline-code">$1</code>');

  // Headers
  html = html.replace(/^#### (.+)$/gm, '<h4 class="md-h4">$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3 class="md-h3">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 class="md-h2">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 class="md-h1">$1</h1>');

  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote class="md-blockquote">$1</blockquote>');

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr class="md-hr" />');

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li class="md-li">$1</li>');
  html = html.replace(
    /(<li class="md-li">[\s\S]*?<\/li>)(\n(?!<li)|\s*$)/g,
    '<ul class="md-ul">$1</ul>$2'
  );

  // Paragraphs: wrap remaining non-tag, non-placeholder lines
  html = html.replace(
    /^(?!<[a-z/]|__CODE_BLOCK_|$)(.+)$/gm,
    '<p class="md-p">$1</p>'
  );

  // Step 2: Restore code blocks from placeholders
  html = html.replace(/__CODE_BLOCK_(\d+)__/g, (_match, idx) => {
    return codeBlocks[parseInt(idx)];
  });

  return html;
}

export default function MarkdownContent(props: MarkdownContentProps) {
  const rendered = createMemo(() => parseMarkdown(props.source));
  let containerRef: HTMLDivElement | undefined;

  createEffect(() => {
    // Re-evaluate rendered() to track it; the innerHTML attribute
    // applies the new content, then we highlight all code blocks.
    rendered();
    if (!containerRef) return;
    const blocks = containerRef.querySelectorAll<HTMLElement>(
      "pre.md-code-block code[class*='language-']",
    );
    blocks.forEach((el) => {
      // hljs marks elements as highlighted; reset to allow re-highlighting
      // when navigating between katas (innerHTML replaces the nodes anyway,
      // so this is mostly defensive).
      delete el.dataset.highlighted;
      hljs.highlightElement(el);
    });
  });

  return (
    <div
      ref={containerRef}
      class="markdown-content"
      innerHTML={rendered()}
    />
  );
}
