#!/usr/bin/env python3
"""Build a DRAFT-PREVIEW PDF of docs/WHITEPAPER.md.

⚠️ This is NOT the canonical published PDF. The canonical PDF at
docs/WHITEPAPER.pdf is typeset with pdfTeX/LaTeX (proper ToC, running
headers, real math symbols, dotted page numbers) and is the version
that should be linked from publications, press, and external citations.

This script's purpose is a quick local rebuild when the markdown changes
and you want a "what would this look like as a PDF?" preview without
running the LaTeX pipeline. Output goes to docs/WHITEPAPER.preview.pdf
to avoid overwriting the canonical artifact.

Pipeline:
  Markdown (with extensions) → HTML → CSS-styled HTML → PDF (via weasyprint)

Mermaid diagrams: rendered via mermaid.ink to inline SVG with on-disk
caching. Falls back to a fenced code block if the service is unreachable.

LaTeX math (single $..$ and block $$..$$) is preserved as-is so a reader
on a math-aware PDF viewer sees the formulae. The LaTeX-typeset canonical
PDF renders math properly; this draft does not.

Output: docs/WHITEPAPER.preview.pdf
"""

from __future__ import annotations

import base64
import re
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import markdown
from weasyprint import CSS, HTML

REPO = Path(__file__).resolve().parent.parent
MD = REPO / "docs" / "WHITEPAPER.md"
OUT = REPO / "docs" / "WHITEPAPER.preview.pdf"

CSS_STR = """
@page {
  size: Letter;
  margin: 0.75in 0.85in;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 9pt;
    color: #666;
  }
  @top-right {
    content: "StateSet Agents — Whitepaper v0.13.4";
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 8.5pt;
    color: #999;
  }
}
@page :first {
  @top-right { content: ""; }
  @bottom-center { content: ""; }
}

body {
  font-family: 'Charter', 'Iowan Old Style', 'Georgia', serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #1a1a1a;
  max-width: 100%;
}

h1 {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 24pt;
  font-weight: 700;
  margin-top: 0.4in;
  margin-bottom: 0.15in;
  color: #0a0a0a;
  page-break-before: always;
}
h1:first-of-type { page-break-before: avoid; }

h2 {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 16pt;
  font-weight: 700;
  margin-top: 0.35in;
  margin-bottom: 0.1in;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.05in;
  color: #0a0a0a;
  page-break-after: avoid;
}

h3 {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 13pt;
  font-weight: 600;
  margin-top: 0.25in;
  margin-bottom: 0.07in;
  color: #1a1a1a;
  page-break-after: avoid;
}

h4 {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 11pt;
  font-weight: 600;
  margin-top: 0.18in;
  margin-bottom: 0.05in;
  color: #2a2a2a;
}

p { margin: 0.07in 0; }

ul, ol { margin: 0.07in 0; padding-left: 0.3in; }
li { margin: 0.03in 0; }

code {
  font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', monospace;
  font-size: 0.88em;
  background-color: #f4f4f4;
  padding: 0.5pt 3pt;
  border-radius: 2pt;
}

pre {
  background-color: #f8f8f8;
  border: 1px solid #e1e1e1;
  border-radius: 3pt;
  padding: 6pt 10pt;
  font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', monospace;
  font-size: 8.5pt;
  line-height: 1.35;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code { background-color: transparent; padding: 0; }

blockquote {
  border-left: 3px solid #4a90e2;
  background-color: #f4f8fc;
  padding: 4pt 10pt;
  margin: 0.1in 0;
  color: #2a2a2a;
  page-break-inside: avoid;
}

table {
  border-collapse: collapse;
  margin: 0.1in 0;
  font-size: 9.5pt;
  width: 100%;
  page-break-inside: auto;
}
table th, table td {
  border: 1px solid #d4d4d4;
  padding: 4pt 7pt;
  text-align: left;
  vertical-align: top;
}
table th {
  background-color: #f0f0f0;
  font-weight: 600;
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}
table tr:nth-child(even) td { background-color: #fafafa; }

hr { border: none; border-top: 1px solid #ccc; margin: 0.25in 0; }

a { color: #0066cc; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Inline math (kept as plain $...$ source — readers with math-aware viewers will see formulae). */
.math { font-family: 'Latin Modern Math', 'STIX Two Math', serif; }

/* Custom callout style for blockquotes that start with an emoji. */
blockquote p:first-child { font-weight: 600; }

/* Cover page */
.cover {
  text-align: center;
  margin-top: 2.5in;
}
.cover .title {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 28pt;
  font-weight: 800;
  line-height: 1.15;
  margin-bottom: 0.4in;
  color: #0a0a0a;
}
.cover .subtitle {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 14pt;
  font-weight: 400;
  color: #444;
  margin-bottom: 0.6in;
}
.cover .version {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 11pt;
  color: #666;
}
.cover .footer {
  position: fixed;
  bottom: 0.85in;
  left: 0;
  right: 0;
  text-align: center;
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-size: 9pt;
  color: #888;
}

/* Mermaid placeholder note (used only when SVG render fails) */
.mermaid-note {
  font-style: italic;
  color: #666;
  font-size: 9pt;
  margin-bottom: 4pt;
}

/* Mermaid SVG rendered inline via mermaid.ink */
.mermaid-rendered {
  margin: 0.15in 0;
  text-align: center;
  page-break-inside: avoid;
}
.mermaid-rendered svg {
  max-width: 100%;
  height: auto;
}
"""


def _mermaid_to_svg(diagram: str, *, timeout: float = 15.0) -> str | None:
    """Render a mermaid diagram via mermaid.ink, return the SVG markup or None on failure.

    Uses the public service's base64-URL endpoint. No JS execution required;
    pure HTTP. We cache results on disk to keep PDF builds reproducible
    even if mermaid.ink is unreachable.
    """
    encoded = (
        base64.urlsafe_b64encode(diagram.encode("utf-8")).decode("ascii").rstrip("=")
    )
    cache_dir = Path(tempfile.gettempdir()) / "mermaid_svg_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{encoded[:32]}.svg"
    if cache_path.exists():
        return cache_path.read_text()
    # URL is always the hardcoded https://mermaid.ink/svg/ endpoint (the
    # diagram content is base64-encoded into the *path*, not the scheme or
    # host) — never attacker-influenced, so the "file:/ or custom scheme"
    # risk this check guards against doesn't apply here.
    url = f"https://mermaid.ink/svg/{encoded}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "stateset-agents-whitepaper-build/0.13.4 (+https://github.com/stateset/stateset-agents)"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: B310
            if resp.status != 200:
                return None
            svg = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as e:
        print(f"  [mermaid] fetch failed for diagram (len={len(diagram)}): {e}")
        return None
    cache_path.write_text(svg, encoding="utf-8")
    return svg


def preprocess_markdown(src: str) -> str:
    """Pre-process the markdown source before it hits the markdown library.

    Mermaid blocks are rendered via mermaid.ink to inline SVG when reachable;
    falls back to a fenced code block + note if the service is unavailable.
    """

    def mermaid_replace(m: re.Match[str]) -> str:
        body = m.group(1)
        svg = _mermaid_to_svg(body)
        if svg is not None:
            # Strip the XML declaration if present so it slots cleanly into HTML.
            svg = re.sub(r"<\?xml[^?]*\?>", "", svg).strip()
            return "<div class='mermaid-rendered'>\n" + svg + "\n</div>\n"
        return (
            "<div class='mermaid-note'>"
            "[Mermaid diagram — render failed; see docs/WHITEPAPER.md on GitHub for the SVG.]"
            "</div>\n\n```\n" + body + "\n```\n"
        )

    return re.sub(
        r"```mermaid\n(.*?)\n```",
        mermaid_replace,
        src,
        flags=re.DOTALL,
    )


def build_cover_html() -> str:
    today = datetime.now(timezone.utc).strftime("%B %Y")
    return f"""
<div class="cover">
  <div class="title">StateSet Agents</div>
  <div class="subtitle">A Reinforcement Learning Framework<br/>for Multi-Turn Conversational AI</div>
  <div class="version">
    <strong>Version 0.13.4</strong> · {today}<br/>
    <a href="mailto:team@stateset.ai">team@stateset.ai</a> · <a href="https://github.com/stateset/stateset-agents">github.com/stateset/stateset-agents</a>
  </div>
  <div class="footer">
    Pinned to source commit. See Appendix C for reproducibility commands.<br/>
    Licensed BUSL-1.1 (transitioning to Apache 2.0 on 2029-09-03).
  </div>
</div>
<div style="page-break-after: always;"></div>
"""


def main() -> int:
    print(f"Reading {MD}...")
    src = MD.read_text()
    src = preprocess_markdown(src)
    print(f"  {len(src):,} chars, {src.count(chr(10)):,} lines")

    md = markdown.Markdown(
        extensions=[
            "extra",  # tables, fenced code, footnotes, etc.
            "toc",  # table of contents anchors
            "sane_lists",
            "admonition",
            "codehilite",
            "nl2br",  # not used but harmless
        ],
        extension_configs={
            "codehilite": {"guess_lang": False, "use_pygments": True},
            "toc": {"baselevel": 1},
        },
    )
    body_html = md.convert(src)
    print(f"  Rendered HTML: {len(body_html):,} chars")

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>StateSet Agents — Whitepaper v0.13.4</title>
</head>
<body>
{build_cover_html()}
{body_html}
</body>
</html>
"""

    print(f"Generating PDF → {OUT}...")
    HTML(string=full_html, base_url=str(REPO)).write_pdf(
        target=str(OUT),
        stylesheets=[CSS(string=CSS_STR)],
    )

    size_kb = OUT.stat().st_size / 1024
    print(f"  ✓ {OUT.relative_to(REPO)} ({size_kb:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
