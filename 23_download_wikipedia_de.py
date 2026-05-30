#!/usr/bin/env python3
"""
Lädt deutsche Wikipedia-Artikel via HuggingFace und extrahiert saubere Sätze.

Quelle:  Wikipedia DE (Wikimedia Foundation)
Lizenz:  CC BY-SA 4.0
         https://creativecommons.org/licenses/by-sa/4.0/

Output: data/wikipedia_de.txt

Usage:
  .venv_ml/bin/python 23_download_wikipedia_de.py [--max-sentences 5000000]
"""

import argparse
import re
from pathlib import Path

from datasets import load_dataset

OUT_FILE = Path("data/wikipedia_de.txt")


def split_sentences(text: str) -> list[str]:
    """Teilt Artikel-Text in Sätze. Einfache Heuristik die für Enzyklopädie-Text gut funktioniert."""
    # Abschnitte/Überschriften entfernen (== Überschrift ==)
    text = re.sub(r"={2,}[^=]+={2,}", " ", text)
    # Mehrfache Leerzeichen/Zeilenumbrüche normalisieren
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)

    # An Satzenden splitten: . ! ? gefolgt von Großbuchstabe
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", text)

    result = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if len(words) < 5 or len(words) > 80:
            continue
        # Keine Sätze mit vielen Sonderzeichen (Tabellen, Vorlagen)
        if len(re.findall(r"[|{}\[\]<>]", s)) > 2:
            continue
        result.append(s)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sentences", type=int, default=5_000_000,
                        help="Maximale Satzanzahl (default: 5M)")
    parser.add_argument("--snapshot", default="20231101.de",
                        help="Wikipedia-Snapshot (default: 20231101.de)")
    args = parser.parse_args()

    print(f"Lade Wikipedia DE ({args.snapshot}) via HuggingFace …")
    ds = load_dataset("wikimedia/wikipedia", args.snapshot,
                      split="train", streaming=True, trust_remote_code=False)

    total = 0
    articles = 0
    OUT_FILE.parent.mkdir(exist_ok=True)

    with OUT_FILE.open("a", encoding="utf-8") as fout:
        for article in ds:
            sentences = split_sentences(article["text"])
            for s in sentences:
                fout.write(s + "\n")
            total += len(sentences)
            articles += 1

            if articles % 10_000 == 0:
                print(f"  {articles:,} Artikel · {total:,} Sätze …")

            if total >= args.max_sentences:
                break

    print(f"\nFertig: {total:,} Sätze aus {articles:,} Artikeln → {OUT_FILE}")


if __name__ == "__main__":
    main()
