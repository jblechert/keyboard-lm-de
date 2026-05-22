#!/usr/bin/env python3
"""
Streams the German Wikipedia bz2 XML dump and writes one sentence per line
to data/train.txt. No wikiextractor dependency — pure stdlib + regex.

Runtime: ~15-30 min for the full dewiki dump.
"""

import bz2
import html
import re
import sys
from pathlib import Path
from xml.etree.ElementTree import iterparse

DUMP = Path("data/dewiki-latest-pages-articles.xml.bz2")
OUT  = Path("data/train.txt")

MIN_CHARS = 20
MAX_CHARS = 500

# ── wikitext cleanup ──────────────────────────────────────────────────────────

def remove_nested(text: str, open_ch: str, close_ch: str) -> str:
    """Remove all spans bounded by open_ch / close_ch, including nested ones."""
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i:i+len(open_ch)] == open_ch:
            depth += 1
            i += len(open_ch)
        elif text[i:i+len(close_ch)] == close_ch:
            if depth > 0:
                depth -= 1
            i += len(close_ch)
        else:
            if depth == 0:
                result.append(text[i])
            i += 1
    return "".join(result)


_FILE_PREFIXES = ('datei:', 'file:', 'bild:', 'image:')
_RE_EXT_LINK  = re.compile(r'\[https?://[^\s\]]*\s*([^\]]*)\]')
_RE_HEADING   = re.compile(r'^={1,6}.+={1,6}\s*$', re.MULTILINE)
_RE_HTML      = re.compile(r'<[^>]+>')
_RE_BOLD_IT   = re.compile(r"'{2,3}")
_RE_WHITESPACE = re.compile(r'[ \t]+')
# Only split at sentence boundaries followed by a German uppercase letter.
_RE_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])')
# Protect periods in abbreviations so they don't trigger sentence splits.
_ABBREV_RE = re.compile(
    r'(?:'
    r'\b(z|bzw|usw|etc|ca|dr|prof|hr|fr|ggf|evtl|vs|vgl|nr|str|'
    r'jan|feb|mär|apr|jun|jul|aug|sep|okt|nov|dez|jh|mio|mrd|'
    r'lt|inkl|exkl|bzgl|abs|art|max|min|u|a|d|h|o|b|e|i|s|v|n)'
    r'|\d+'   # ordinal numbers: 3. Januar, 21. Mai
    r')\.',
    re.IGNORECASE
)
_PLACEHOLDER = '\x00'


def process_wikilinks(text: str) -> str:
    """
    Walk through text handling [[...]] with proper nesting.
    - File/Image/Datei/Bild links → removed entirely (incl. nested content)
    - Regular links → replaced with display text (last pipe segment)
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+2] == '[[':
            # Find matching ]] respecting nesting
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text[j:j+2] == '[[':
                    depth += 1
                    j += 2
                elif text[j:j+2] == ']]':
                    depth -= 1
                    j += 2
                else:
                    j += 1
            # Unmatched [[ (e.g. chemical/math notation) — treat as literal
            if depth > 0:
                result.append('[[')
                i += 2
                continue
            inner = text[i+2:j-2]
            # Determine first segment (before first |) for prefix check
            pipe = inner.find('|')
            first_raw = inner[:pipe] if pipe != -1 else inner
            first = first_raw.strip()
            if (first.lower().startswith(_FILE_PREFIXES)
                    or not first                              # empty target
                    or first_raw and first_raw[0] in (' ', '\n', '\t')):  # [[ action]] links
                pass  # drop entirely
            else:
                # Keep last pipe segment as display text; strip residual markup
                display = (inner[pipe+1:] if pipe != -1 else inner).strip()
                display = re.sub(r'\[\[|\]\]', '', display)
                result.append(display)
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def clean(wikitext: str) -> str:
    t = wikitext
    # Remove <ref>...</ref> blocks (footnotes)
    t = re.sub(r'<ref[^>]*/>', '', t)
    t = re.sub(r'<ref[^>]*>.*?</ref>', '', t, flags=re.DOTALL)
    # Remove templates {{...}} including nested
    t = remove_nested(t, '{{', '}}')
    # Remove tables {|...|}
    t = re.sub(r'\{\|.*?\|\}', '', t, flags=re.DOTALL)
    # Resolve wikilinks — keep display text, drop file/image links
    t = process_wikilinks(t)
    # Remove external links
    t = _RE_EXT_LINK.sub(r'\1', t)
    # Remove headings
    t = _RE_HEADING.sub('', t)
    # Remove remaining HTML tags
    t = _RE_HTML.sub('', t)
    # Remove bold/italic markers
    t = _RE_BOLD_IT.sub('', t)
    # Decode HTML entities (&nbsp; &amp; etc.) and normalize non-breaking spaces
    t = html.unescape(t).replace('\xa0', ' ')
    # Collapse whitespace per line
    lines = (_RE_WHITESPACE.sub(' ', line).strip() for line in t.splitlines())
    # Drop lines that are clearly list items, file links, categories, etc.
    lines = (
        l for l in lines
        if l
        and not l.startswith(('*', '#', ':', ';', '|', '!'))
        and not l.lower().startswith(('datei:', 'file:', 'bild:', 'image:',
                                       'kategorie:', 'category:'))
    )
    return ' '.join(lines)


def sentences(text: str):
    # Protect abbreviation-periods before splitting
    protected = _ABBREV_RE.sub(
        lambda m: m.group(0)[:-1] + _PLACEHOLDER, text
    )
    for s in _RE_SENTENCE_END.split(protected):
        s = s.replace(_PLACEHOLDER, '.').strip()
        if MIN_CHARS <= len(s) <= MAX_CHARS:
            yield s


# ── XML streaming ─────────────────────────────────────────────────────────────

NS = '{http://www.mediawiki.org/xml/export-0.11/}'

def iter_article_texts(dump_path: Path):
    """Yield raw wikitext for every article (namespace 0) in the dump."""
    with bz2.open(dump_path, 'rb') as f:
        in_article = False
        title = ''
        for event, elem in iterparse(f, events=('start', 'end')):
            tag = elem.tag

            if event == 'start' and tag == f'{NS}page':
                in_article = False
                title = ''

            elif event == 'end' and tag == f'{NS}ns':
                in_article = (elem.text == '0')

            elif event == 'end' and tag == f'{NS}title':
                title = elem.text or ''

            elif event == 'end' and tag == f'{NS}text':
                if in_article and elem.text:
                    text = elem.text
                    # Skip redirects
                    if not text.lstrip().upper().startswith('#REDIRECT'):
                        yield text
                elem.clear()

            elif event == 'end' and tag == f'{NS}page':
                elem.clear()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not DUMP.exists():
        print(f"Dump not found: {DUMP}", file=sys.stderr)
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    articles = kept = 0
    with OUT.open('w', encoding='utf-8') as out:
        for wikitext in iter_article_texts(DUMP):
            articles += 1
            text = clean(wikitext)
            for s in sentences(text):
                out.write(s + '\n')
                kept += 1
            if articles % 10_000 == 0:
                print(f"  {articles:,} Artikel verarbeitet, {kept:,} Sätze …", flush=True)

    print(f"\nFertig: {articles:,} Artikel → {kept:,} Sätze in {OUT}")


if __name__ == '__main__':
    main()
