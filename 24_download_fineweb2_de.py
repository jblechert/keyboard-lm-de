#!/usr/bin/env python3
"""
Streamt deutsche Texte aus FineWeb2-HQ und extrahiert einzelne Sätze.

Quelle:  FineWeb2-HQ (HuggingFace FineData)
         https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
Lizenz:  ODC-By v1.0 (Open Data Commons Attribution)
         https://opendatacommons.org/licenses/by/1-0/
         Namensnennung: HuggingFace FineData / CommonCrawl

Konfiguration: deu_Latn (Deutsch)

Output: data/fineweb2_de.txt

Usage:
  .venv_ml/bin/python 24_download_fineweb2_de.py [--target 80000000] [--resume]
"""

import argparse
import re
import sys
from pathlib import Path

OUT_FILE = Path("data/fineweb2_de.txt")

SPLIT_PAT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ\"\„\[])')
MIN_WORDS = 5
MAX_WORDS = 60


def split_document(text: str) -> list[str]:
    """Teilt einen Dokument-Text in einzelne Sätze."""
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    sentences = []
    for paragraph in text.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = SPLIT_PAT.split(paragraph)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            words = len(part.split())
            if words < MIN_WORDS or words > MAX_WORDS:
                continue
            if not re.search(r'[.!?]$', part):
                part = part  # Kein Punkt nötig, Satz trotzdem nützlich
            sentences.append(part)
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=80_000_000,
                        help="Ziel-Satzanzahl (default: 80M)")
    parser.add_argument("--resume", action="store_true",
                        help="Weitermachen wenn Datei bereits existiert")
    args = parser.parse_args()

    # Bereits vorhandene Sätze zählen
    existing = 0
    if OUT_FILE.exists():
        if not args.resume:
            print(f"Datei existiert bereits: {OUT_FILE}")
            print("Nutze --resume um weiterzumachen oder lösche die Datei.")
            sys.exit(1)
        existing = sum(1 for _ in OUT_FILE.open(encoding="utf-8"))
        print(f"Fortsetze ab {existing:,} vorhandenen Sätzen …")

    if existing >= args.target:
        print(f"Ziel bereits erreicht: {existing:,} ≥ {args.target:,}")
        sys.exit(0)

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets nicht installiert: pip install datasets")
        sys.exit(1)

    print(f"Streame FineWeb2-HQ deu_Latn → {OUT_FILE}")
    print(f"Ziel: {args.target:,} Sätze (bereits {existing:,} vorhanden)\n")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        "deu_Latn",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    total = existing
    docs_processed = 0
    milestone = existing + 1_000_000

    with OUT_FILE.open("a", encoding="utf-8") as fout:
        for doc in ds:
            sentences = split_document(doc["text"])
            for s in sentences:
                fout.write(s + "\n")
            total += len(sentences)
            docs_processed += 1

            if total >= milestone:
                fout.flush()
                print(f"  {total/1_000_000:.1f}M Sätze ({docs_processed:,} Dokumente)", flush=True)
                milestone = total + 1_000_000

            if total >= args.target:
                break

    print(f"\nFertig: {total:,} Sätze → {OUT_FILE}")
    print(f"Dokumente verarbeitet: {docs_processed:,}")


if __name__ == "__main__":
    main()
