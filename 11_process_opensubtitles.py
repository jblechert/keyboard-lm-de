#!/usr/bin/env python3
"""
Verarbeitet die rohen OpenSubtitles-Zeilen zu sauberem Trainingstext.

Input:  data/opensubtitles/de.txt.gz
Output: data/opensubtitles/de_clean.txt

OpenSubtitles-Besonderheiten:
  - Untertitel-Zeilen, oft ohne Satzzeichen am Ende
  - Sehr kurze Zeilen (Wörter, Ausrufe)
  - Regie-Anweisungen in Klammern: (lacht), [Musik]
  - Kursiv-Markup: <i>Text</i>
  - Nummerierungen und leere Zeilen
  - Viel Wiederholung (gleiche Untertitel in vielen Filmen)
"""

import gzip
import hashlib
import html
import re
import sys
from pathlib import Path

IN_GZ   = Path("data/opensubtitles/de.txt.gz")
OUT_TXT = Path("data/opensubtitles/de_clean.txt")

MIN_CHARS = 15
MAX_CHARS = 300

_RE_TAG      = re.compile(r"<[^>]+>")
_RE_BRACKET  = re.compile(r"[\(\[{][^\)\]\}]{0,40}[\)\]\}]")  # (lacht), [Musik]
_RE_DASH_PRE = re.compile(r"^\s*[-–—]+\s*")                   # - Dialog-Dashes
_RE_DOTS     = re.compile(r"\.{2,}")                           # ... → .
_RE_SPACES   = re.compile(r"\s+")
_RE_NUMBERS  = re.compile(r"^\d+$")                            # reine Nummern (Index)


def clean(line: str) -> str:
    t = html.unescape(line.strip())
    t = _RE_TAG.sub("", t)           # HTML-Tags
    t = _RE_BRACKET.sub("", t)       # Regieanweisungen
    t = _RE_DASH_PRE.sub("", t)      # Dialog-Dashes am Anfang
    t = _RE_DOTS.sub(".", t)         # Ellipsen normalisieren
    t = _RE_SPACES.sub(" ", t).strip()
    return t


def is_garbage(line: str) -> bool:
    if not line:
        return True
    if _RE_NUMBERS.match(line):      # reine Indexnummern
        return True
    if line.isupper() and len(line) < 20:  # SCHREIEN / TITELKARTEN
        return True
    # Zu viele Sonderzeichen → kein natürlicher Text
    special = sum(1 for c in line if not c.isalnum() and c not in " .,!?-–':\"")
    if special / len(line) > 0.3:
        return True
    return False


def main():
    if not IN_GZ.exists():
        print(f"Nicht gefunden: {IN_GZ} — erst 10_download_opensubtitles.sh ausführen.", file=sys.stderr)
        sys.exit(1)

    seen: set[str] = set()   # Deduplizierung via MD5
    kept = skipped = dupes = 0

    with gzip.open(IN_GZ, "rt", encoding="utf-8", errors="replace") as f_in, \
         OUT_TXT.open("w", encoding="utf-8") as f_out:

        for raw in f_in:
            line = clean(raw)

            if not (MIN_CHARS <= len(line) <= MAX_CHARS):
                skipped += 1
                continue
            if is_garbage(line):
                skipped += 1
                continue

            # Deduplizierung
            h = hashlib.md5(line.encode()).digest()
            if h in seen:
                dupes += 1
                continue
            seen.add(h)

            f_out.write(line + "\n")
            kept += 1

            if kept % 500_000 == 0:
                print(f"  {kept:,} Zeilen behalten ...", flush=True)

    print(f"\nFertig:")
    print(f"  Behalten:      {kept:,}")
    print(f"  Gefiltert:     {skipped:,}")
    print(f"  Duplikate:     {dupes:,}")
    print(f"  Ausgabe:       {OUT_TXT}")


if __name__ == "__main__":
    main()
