#!/usr/bin/env python3
"""
Generiert eine Bannliste aller 3-4-Buchstaben-Sequenzen (a-z) die keine
deutschen Wörter sind. Abfrage via Ollama/Qwen3.

Vorgehen:
  1. Alle 26^3 + 26^4 = 474.552 Kombinationen generieren
  2. In 400er-Batches an Qwen3 schicken: "Welche davon sind deutsche Wörter?"
  3. Checkpoint nach je 10 Batches → unterbrechbar/fortsetzbar
  4. Corpus-Scan: nur Tokens die auch in c4_de.txt vorkommen → finale Bannliste

Ausgabe:
  data/garbled_german_words.json   (Checkpoint + deutsche Wörter)
  data/garbled_banlist.txt         (finale Bannliste, ein Token pro Zeile)

Usage:
  python build_garbled_banlist.py
  python build_garbled_banlist.py --corpus-only   (überspringt Qwen3, nur Corpus-Scan)
"""
import argparse
import itertools
import json
import re
import time
import urllib.request
from pathlib import Path

OLLAMA_URL   = "http://localhost:11434/api/chat"
MODEL        = "qwen3.6:27b"
BATCH_SIZE   = 400
PROGRESS_FILE = Path("data/garbled_german_words.json")
BAN_LIST_FILE = Path("data/garbled_banlist.txt")
CORPUS_FILES  = [Path("data/c4_de.txt"), Path("data/fineweb2_de.txt")]


def query_german(words: list[str], retries: int = 3) -> set[str]:
    """Fragt Qwen3 welche Zeichenfolgen aus `words` gültige deutsche Wörter sind."""
    prompt = (
        "/no_think\n"
        "Antworte NUR mit einem JSON-Array (kein weiterer Text, keine Erklärung).\n"
        "Welche der folgenden Zeichenfolgen sind gültige deutsche Wörter? "
        "(Präpositionen, Konjunktionen, Pronomen, Verbformen, Abkürzungen "
        "die im Deutschen geläufig sind, Kurzwörter — alles zählt)\n\n"
        f"{json.dumps(words, ensure_ascii=False)}"
    )
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 2048},
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                text = json.loads(r.read())["message"]["content"]

            # Qwen3 thinking-Blöcke entfernen
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            start = text.find("[")
            end   = text.rfind("]") + 1
            if start == -1 or end == 0:
                return set()
            return {w.lower() for w in json.loads(text[start:end]) if isinstance(w, str)}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠ Fehler nach {retries} Versuchen: {e}")
                return set()
    return set()


def corpus_scan(german_words: set[str]) -> list[str]:
    """Scannt Corpus-Dateien und gibt nicht-deutsche Tokens zurück die vorkommen."""
    corpus_tokens: set[str] = set()
    for path in CORPUS_FILES:
        if not path.exists():
            continue
        print(f"Scanne {path.name} ...")
        with path.open(encoding="utf-8", errors="ignore") as f:
            for n, line in enumerate(f, 1):
                for tok in re.findall(r"\b[a-z]{3,4}\b", line):
                    corpus_tokens.add(tok)
                if n % 2_000_000 == 0:
                    print(f"  {n // 1_000_000}M Zeilen | {len(corpus_tokens):,} Tokens")
        print(f"  fertig: {len(corpus_tokens):,} unique Tokens in {path.name}")

    all_combos = set(
        "".join(c)
        for length in (3, 4)
        for c in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=length)
    )
    non_german_in_corpus = sorted((all_combos - german_words) & corpus_tokens)
    print(f"Nicht-deutsch im Corpus: {len(non_german_in_corpus):,} Tokens")
    return non_german_in_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-only", action="store_true",
                        help="Qwen3-Phase überspringen, nur Corpus-Scan")
    args = parser.parse_args()

    # ── Alle Kombinationen ────────────────────────────────────────────────────
    all_combos: list[str] = []
    for length in (3, 4):
        for combo in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=length):
            all_combos.append("".join(combo))

    total = len(all_combos)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"{total:,} Kombinationen → {n_batches} Batches à {BATCH_SIZE}")

    # ── Fortschritt laden ─────────────────────────────────────────────────────
    german_words: set[str] = set()
    done_batches = 0
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        german_words = set(data.get("german_words", []))
        done_batches = data.get("done_batches", 0)
        print(f"Checkpoint: {done_batches}/{n_batches} Batches, "
              f"{len(german_words):,} deutsche Wörter bekannt")

    # ── Qwen3-Phase ───────────────────────────────────────────────────────────
    if not args.corpus_only:
        t0 = time.time()
        processed = 0

        for i in range(0, total, BATCH_SIZE):
            batch_idx = i // BATCH_SIZE
            if batch_idx < done_batches:
                continue

            batch = all_combos[i : i + BATCH_SIZE]
            found = query_german(batch)
            german_words.update(found)
            processed += 1

            elapsed  = time.time() - t0
            per_batch = elapsed / processed
            remaining = (n_batches - batch_idx - 1) * per_batch

            if processed % 10 == 0 or processed <= 3:
                PROGRESS_FILE.write_text(json.dumps(
                    {"german_words": sorted(german_words), "done_batches": batch_idx + 1},
                    ensure_ascii=False, indent=2,
                ))
                print(
                    f"  Batch {batch_idx+1:5d}/{n_batches}"
                    f" | +{len(found):3d} deutsch"
                    f" | gesamt {len(german_words):,}"
                    f" | {per_batch:.1f}s/Batch"
                    f" | ETA {remaining/60:.1f} min"
                )

        # Finaler Checkpoint
        PROGRESS_FILE.write_text(json.dumps(
            {"german_words": sorted(german_words), "done_batches": n_batches},
            ensure_ascii=False, indent=2,
        ))
        print(f"\nQwen3-Phase fertig: {len(german_words):,} deutsche Wörter identifiziert")

    # ── Corpus-Scan ───────────────────────────────────────────────────────────
    if not german_words and PROGRESS_FILE.exists():
        german_words = set(json.loads(PROGRESS_FILE.read_text()).get("german_words", []))

    print()
    ban_list = corpus_scan(german_words)

    BAN_LIST_FILE.write_text("\n".join(ban_list) + "\n")
    print(f"\n→ {BAN_LIST_FILE}  ({len(ban_list):,} Tokens)")

    print("\nErste 80 Tokens aus der Bannliste:")
    for w in ban_list[:80]:
        print(f"  {w}", end="")
    print()


if __name__ == "__main__":
    main()
