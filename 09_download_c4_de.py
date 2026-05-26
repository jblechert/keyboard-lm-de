#!/usr/bin/env python3
"""
Streams German sentences from allenai/c4 (mC4, ODC-BY license),
splits documents into sentences, filters for keyboard-appropriate length,
and saves to data/c4_de.txt.

Output: data/c4_de.txt  (one sentence per line)

Usage:
  .venv_ml/bin/python 09_download_c4_de.py [--target 5000000]
"""

import argparse
import re
import sys
import time
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Fehler: datasets-Paket fehlt.", file=sys.stderr)
    print("  .venv_ml/bin/pip install datasets", file=sys.stderr)
    sys.exit(1)

OUT = Path("data/c4_de.txt")

# Split on sentence boundaries AND newlines (c4 docs have mixed line structure)
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])|[\n\r]+')

# Quick filters
_URL_RE    = re.compile(r'https?://\S+|www\.\S+')
_EMAIL_RE  = re.compile(r'\S+@\S+\.\S+')
_ENDS_NUM  = re.compile(r'\s\d+\.\s*$')  # truncated: "... am 15."
_BAD_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]|--')  # encoding junk
_NAV_RE    = re.compile(r'^[A-ZÄÖÜ][^.!?]{0,40} - [A-ZÄÖÜ]')  # "Home - Homepage..."

# German-specific words (umlauts or words that don't exist in English)
# Requires at least one unambiguously German word
_DE_WORDS  = re.compile(
    r'[äöüÄÖÜß]|'
    r'\b(ist|sind|war|waren|wird|werden|hat|haben|hatte|hatten|'
    r'ich|nicht|auch|noch|aber|oder|dass|wenn|weil|als|welche|welcher|'
    r'bei|für|durch|nach|über|unter|vor|beim|zum|zur|vom|ins|'
    r'werden|wurde|wurden|können|müssen|sollen|wollen|dürfen|möchten)\b',
    re.IGNORECASE
)

# Spam/boilerplate patterns
_SPAM_RE   = re.compile(
    r'casino|viagra|porn|sex|xxx|click here|buy now|jetzt kaufen|'
    r'provera|dosierung|mg preis|apotheke|'
    r'http|©|\||\t|<[a-z]|\{|\}|=|\.\.\.',
    re.IGNORECASE
)


def split_sentences(text: str) -> list[str]:
    return _SENT_RE.split(text.strip())


def is_good(sent: str) -> bool:
    words = sent.split()
    n = len(words)
    if n < 5 or n > 25:
        return False
    if _URL_RE.search(sent) or _EMAIL_RE.search(sent):
        return False
    if _BAD_CHARS.search(sent):
        return False
    if _NAV_RE.match(sent):
        return False
    if _SPAM_RE.search(sent):
        return False
    # Truncated sentence ending in a bare number: "... am 15."
    if _ENDS_NUM.search(sent):
        return False
    # Must contain at least one German function word
    if not _DE_WORDS.search(sent):
        return False
    # Skip lines that are mostly digits
    digit_ratio = sum(1 for c in sent if c.isdigit()) / max(len(sent), 1)
    if digit_ratio > 0.12:
        return False
    # Skip all-caps lines
    letters = [c for c in sent if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.5:
        return False
    # Must end with sentence-ending punctuation or a regular word
    if sent[-1] in ',;:–-(':
        return False
    # Garbled text tends to have very short average word length
    avg_word_len = sum(len(w) for w in words) / n
    if avg_word_len < 3.5:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5_000_000,
                        help="Anzahl Sätze (default: 5.000.000)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bestehende Datei überschreiben")
    args = parser.parse_args()

    mode = "w" if args.overwrite else "a"
    existing = 0
    if not args.overwrite and OUT.exists():
        existing = sum(1 for _ in OUT.open())
        if existing >= args.target:
            print(f"Bereits {existing:,} Sätze vorhanden, nichts zu tun.")
            return
        print(f"Setze fort: {existing:,} Sätze vorhanden, Ziel {args.target:,}")

    print(f"Streame allenai/c4 (de) → {OUT}")
    print(f"Ziel: {args.target - existing:,} weitere Sätze\n")

    ds = load_dataset("allenai/c4", "de", streaming=True, split="train",
                      trust_remote_code=False)

    collected = 0
    docs_seen = 0
    t_start = time.time()

    with OUT.open(mode, encoding="utf-8") as f:
        for row in ds:
            docs_seen += 1
            for sent in split_sentences(row["text"]):
                sent = sent.strip()
                if not is_good(sent):
                    continue
                f.write(sent + "\n")
                collected += 1

                if collected % 50_000 == 0:
                    f.flush()
                    elapsed = time.time() - t_start
                    total = existing + collected
                    rate = collected / elapsed
                    eta = (args.target - total) / rate if rate > 0 else 0
                    print(f"  [{total:>9,}/{args.target:,}]  "
                          f"{rate:,.0f} Sätze/s  "
                          f"ETA: {eta/60:.0f} min  "
                          f"({docs_seen:,} Dokumente)")

                if existing + collected >= args.target:
                    break
            if existing + collected >= args.target:
                break

    total = time.time() - t_start
    final = existing + collected
    print(f"\nFertig: {final:,} Sätze gesamt in {total/60:.1f} min → {OUT}")
    print(f"Dateigröße: {OUT.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
