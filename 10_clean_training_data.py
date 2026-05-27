#!/usr/bin/env python3
"""
Bereinigt Trainingsdaten von unerwünschten Wörtern/Phrasen.

Füge neue unerwünschte Muster zur BANNED-Liste hinzu und führe das Skript
erneut aus — es filtert alle gefundenen Quelldateien in-place.

Unterstützt auch --high-ppl FILE: Liest eine Datei mit Sätzen (eine pro Zeile)
und entfernt exakt diese Sätze aus den Quelldateien.

Usage:
  .venv_ml/bin/python 10_clean_training_data.py [--dry-run]
  .venv_ml/bin/python 10_clean_training_data.py --high-ppl data/high_ppl.txt [--dry-run]
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ── Unerwünschte Muster ───────────────────────────────────────────────────────
# Jeder Eintrag: (pattern, beschreibung)
# Groß-/Kleinschreibung wird ignoriert (re.IGNORECASE).
# Ganzes Wort via \b oder freier Substring — je nach Eintrag.

BANNED = [
    # Überrepräsentierte Einzelwörter
    (r"\bmahd\b",       "Mahd (fälschlicherweise häufig vorhergesagt)"),

    # Nicht-lateinische Schriftsysteme (eindeutig nicht Deutsch)
    (r"[Ѐ-ӿ]",    "Kyrillisch"),
    (r"[؀-ۿ]",    "Arabisch"),
    (r"[一-鿿]",    "CJK-Zeichen (Chinesisch/Japanisch)"),
    (r"[가-힯]",    "Hangul (Koreanisch)"),
    (r"[ऀ-ॿ]",    "Devanagari"),

    # Alte deutsche Rechtschreibung / Scanfehler (typisch für schlecht gefilterte Webdaten)
    (r"\bae(?:hn|uss|uss)\w*", "Alte Schreibweise ae- (Aehnliche, Aeusserst …)"),
    (r"\bue(?:ber|brig|bel)\w*", "Alte Schreibweise ue- (Ueber, Uebrig …)"),

    # Spam-Muster: viele Pipe- oder Tabulator-getrennten Felder → Datentabellen
    (r"(?:\||\t){3,}",      "Tabellenzeile (≥3 Pipes/Tabs) — kein Fließtext"),

    # Sehr viele aufeinanderfolgende Sonderzeichen → Listenspam
    (r"[*•·]{4,}",          "Bullet-Spam (≥4 Sonderzeichen in Folge)"),
]

# ── Quelldateien ──────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/tatoeba_de.txt"),
    Path("data/c4_de.txt"),
    *sorted(Path("data").glob("synthetic_*.txt")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur zählen, nichts schreiben")
    parser.add_argument("--high-ppl", metavar="FILE",
                        help="Datei mit Sätzen hoher Perplexity (eine pro Zeile); "
                             "exakte Übereinstimmungen werden entfernt")
    args = parser.parse_args()

    patterns = [(re.compile(p, re.IGNORECASE), desc) for p, desc in BANNED]

    # Optionale Menge von Sätzen aus --high-ppl
    high_ppl_set: set[str] = set()
    if args.high_ppl:
        ppl_path = Path(args.high_ppl)
        if not ppl_path.exists():
            print(f"Fehler: --high-ppl Datei nicht gefunden: {ppl_path}", file=sys.stderr)
            sys.exit(1)
        high_ppl_set = {l.strip() for l in ppl_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"{len(high_ppl_set)} Sätze aus {ppl_path} geladen (hohe Perplexity).")

    total_removed = 0

    for src in SOURCES:
        if not src.exists():
            continue

        tmp = src.with_suffix(".tmp")
        removed_counts = {desc: 0 for _, desc in BANNED}
        n_ppl_removed = 0
        n_kept = 0
        n_total = 0

        with src.open("r", encoding="utf-8") as fin, \
             (tmp.open("w", encoding="utf-8") if not args.dry_run else open(os.devnull, "w")) as fout:
            for line in fin:
                n_total += 1
                stripped = line.strip()

                if stripped in high_ppl_set:
                    n_ppl_removed += 1
                    continue

                drop = False
                for pat, desc in patterns:
                    if pat.search(line):
                        removed_counts[desc] += 1
                        drop = True
                        break

                if not drop:
                    fout.write(line)
                    n_kept += 1

                if n_total % 5_000_000 == 0:
                    print(f"  {src.name}: {n_total:,} gelesen …", flush=True)

        n_removed = n_total - n_kept
        total_removed += n_removed

        if n_removed == 0:
            print(f"{src}: keine Treffer ({n_total:,} Sätze)")
            if not args.dry_run:
                tmp.unlink()
            continue

        print(f"{src}: {n_removed:,} Sätze entfernt (von {n_total:,})")
        if n_ppl_removed:
            print(f"  {n_ppl_removed:>8,}×  hohe Perplexity (--high-ppl)")
        for desc, count in removed_counts.items():
            if count:
                print(f"  {count:>8,}×  {desc}")

        if not args.dry_run:
            tmp.replace(src)

    print(f"\nGesamt entfernt: {total_removed:,}")
    if args.dry_run:
        print("(Dry-run — keine Datei verändert)")


if __name__ == "__main__":
    main()
