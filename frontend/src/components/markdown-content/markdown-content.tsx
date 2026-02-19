import { createMemo } from "solid-js";
import "./markdown-content.css";

interface MarkdownContentProps {
  source: string;
}

function parseMarkdown(md: string): string {
  let html = md;

  // Code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, lang, code) => {
    const escaped = code
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return `<pre class="md-code-block"><code class="language-${lang}">${escaped}</code></pre>`;
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

  // Paragraphs: wrap remaining non-tag lines
  html = html.replace(
    /^(?!<[a-z/]|$)(.+)$/gm,
    '<p class="md-p">$1</p>'
  );

  return html;
}

export default function MarkdownContent(props: MarkdownContentProps) {
  const rendered = createMemo(() => parseMarkdown(props.source));

  return <div class="markdown-content" innerHTML={rendered()} />;
}
