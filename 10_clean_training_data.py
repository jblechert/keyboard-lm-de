#!/usr/bin/env python3
"""
Bereinigt Trainingsdaten von unerwünschten Wörtern/Phrasen.

Füge neue unerwünschte Muster zur BANNED-Liste hinzu und führe das Skript
erneut aus — es filtert alle gefundenen Quelldateien in-place.

Usage:
  .venv_ml/bin/python 10_clean_training_data.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

# ── Unerwünschte Muster ───────────────────────────────────────────────────────
# Jeder Eintrag: (pattern, beschreibung)
# Groß-/Kleinschreibung wird ignoriert (re.IGNORECASE).
# Ganzes Wort via \b oder freier Substring — je nach Eintrag.

BANNED = [
    (r"\bmahd\b",       "Mahd/Mahdi (fälschlicherweise häufig vorhergesagt)"),
]

# ── Quelldateien ──────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/tatoeba_de.txt"),
    Path("data/c4_de.txt"),
    Path("data/synthetic_de.txt"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur zählen, nichts schreiben")
    args = parser.parse_args()

    patterns = [(re.compile(p, re.IGNORECASE), desc) for p, desc in BANNED]

    total_removed = 0

    for src in SOURCES:
        if not src.exists():
            continue

        lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = []
        removed_counts = {desc: 0 for _, desc in BANNED}

        for line in lines:
            drop = False
            for pat, desc in patterns:
                if pat.search(line):
                    removed_counts[desc] += 1
                    drop = True
                    break
            if not drop:
                kept.append(line)

        n_removed = len(lines) - len(kept)
        total_removed += n_removed

        if n_removed == 0:
            print(f"{src}: keine Treffer")
            continue

        print(f"{src}: {n_removed} Sätze entfernt")
        for desc, count in removed_counts.items():
            if count:
                print(f"  {count:>6}×  {desc}")

        if not args.dry_run:
            src.write_text("".join(kept), encoding="utf-8")

    print(f"\nGesamt entfernt: {total_removed}")
    if args.dry_run:
        print("(Dry-run — keine Datei verändert)")


if __name__ == "__main__":
    main()
