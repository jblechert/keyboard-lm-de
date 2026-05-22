#!/usr/bin/env python3
"""
Sanity checks for the trained SentencePiece tokenizer.
Run after 04_train_tokenizer.py. Prints PASS/FAIL for each check.
Exit code 0 = all checks passed, 1 = at least one failure.
"""

import sys
from pathlib import Path
import sentencepiece as spm

SP_MODEL  = Path("data/tokenizer/de_keyboard.model")
TRAIN_TXT = Path("data/train.txt")

FUTO_SPECIAL_TOKENS = [
    "<XBU>", "<XBC>", "<XEC>",
    *[f"<CHAR_{chr(c)}>" for c in range(ord("A"), ord("Z") + 1)],
]

# Häufige deutsche Wörter → sollten Einzeltokens sein
COMMON_SINGLE_TOKEN_WORDS = [
    "der", "die", "das", "und", "ist", "in", "von", "mit",
    "zu", "den", "an", "auf", "für", "nicht", "sich", "auch",
    "er", "sie", "es", "als", "ich", "du", "wir", "hat",
]

# Umlaute → dürfen nicht als Byte-Sequenzen (<0xXX>) auftauchen
UMLAUT_TEST = "Ärger Öl Über Höhe schön wäre Straße"

# Vollständige Sätze für Roundtrip
ROUNDTRIP_TESTS = [
    "Das Modell soll deutsche Sprache verstehen.",
    "Österreich und die Schweiz sprechen auch Deutsch.",
    "Ärzte, Ökonomen und Übersetzer arbeiten zusammen.",
    "Ein Satz mit, Satzzeichen! Und einem Ausrufezeichen?",
    "Bundesverfassungsgericht und Kraftfahrzeugsteuer sind Komposita.",
]

# Für Suffix-Check: Leerzeichen muss am ENDE des Tokens sein (FUTO-Format)
SUFFIX_TEST_WORD = "Sprache"  # tokenisiert als "Sprache▁" wenn korrekt


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f"  → {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def main():
    if not SP_MODEL.exists():
        print(f"Tokenizer nicht gefunden: {SP_MODEL}")
        print("Erst 04_train_tokenizer.py ausführen.")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    vocab_size = sp.get_piece_size()
    failures = 0

    # ── 1. Vokabulargröße ─────────────────────────────────────────────────────
    print("\n── 1. Vokabulargröße")
    ok = check("Vokabular ≥ 14000 Tokens", vocab_size >= 14000, f"{vocab_size} Tokens")
    failures += not ok
    ok = check("Vokabular ≤ 16000 Tokens", vocab_size <= 16000, f"{vocab_size} Tokens")
    failures += not ok

    # ── 2. Special Tokens ─────────────────────────────────────────────────────
    print("\n── 2. FUTO Special Tokens")
    for tok in FUTO_SPECIAL_TOKENS:
        tid = sp.piece_to_id(tok)
        ok = check(f"{tok} vorhanden", tid > 0, f"id={tid}")
        failures += not ok

    # ── 3. Whitespace-Suffix ──────────────────────────────────────────────────
    print("\n── 3. Whitespace-Suffix (treat_whitespace_as_suffix=true)")
    # Tokenisiere "X Y" → das erste Token für X sollte NICHT mit ▁ beginnen
    # und das zweite Token für " " sollte am Ende stehen oder X▁ sein
    pieces = sp.encode("Sprache Modell", out_type=str)
    has_suffix = any(p.endswith("▁") or p.endswith("▁") for p in pieces)
    has_prefix = any(p.startswith("▁") or p.startswith("▁") for p in pieces)
    ok = check("Leerzeichen als Suffix (▁ am Ende)", has_suffix, str(pieces[:4]))
    failures += not ok
    ok = check("Kein Leerzeichen-Präfix (▁ am Anfang)", not has_prefix, str(pieces[:4]))
    failures += not ok

    # ── 4. Umlaute ────────────────────────────────────────────────────────────
    print("\n── 4. Umlauts & Sonderzeichen")
    pieces = sp.encode(UMLAUT_TEST, out_type=str)
    byte_tokens = [p for p in pieces if p.startswith("<0x")]
    ok = check("Keine Byte-Fallbacks für Umlaute", len(byte_tokens) == 0,
               f"Byte-Tokens gefunden: {byte_tokens}" if byte_tokens else "")
    failures += not ok

    # ── 5. Häufige Wörter als Einzeltokens ───────────────────────────────────
    print("\n── 5. Häufige Wörter (sollten Einzeltokens sein)")
    multi = []
    for word in COMMON_SINGLE_TOKEN_WORDS:
        # Mit Leerzeichen am Ende (wie im Fließtext)
        pieces = sp.encode(word + " ", out_type=str)
        if len(pieces) > 2:   # 1 Worttoken + ggf. Suffix = ok; mehr = Problem
            multi.append((word, pieces))
    ok = check("Häufige Wörter max. 2 Stücke", len(multi) == 0,
               f"Probleme: {[(w, p) for w,p in multi[:3]]}" if multi else "")
    failures += not ok

    # ── 6. Roundtrip ──────────────────────────────────────────────────────────
    print("\n── 6. Encode→Decode Roundtrip (verlustfrei)")
    for text in ROUNDTRIP_TESTS:
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        ok = check(f"Roundtrip: {text[:45]!r}", decoded == text,
                   f"got: {decoded!r}" if decoded != text else "")
        failures += not ok

    # ── 7. Tokens-pro-Wort Statistik ─────────────────────────────────────────
    print("\n── 7. Tokens-pro-Wort Statistik (Stichprobe 10.000 Sätze)")
    if TRAIN_TXT.exists():
        total_tokens = total_words = 0
        with TRAIN_TXT.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10_000:
                    break
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                total_words += len(words)
                total_tokens += len(sp.encode(line))
        ratio = total_tokens / total_words if total_words else 0
        ok = check("Tokens/Wort zwischen 1.1 und 2.5", 1.1 <= ratio <= 2.5,
                   f"{ratio:.2f} Tokens/Wort ({total_tokens:,} Tokens / {total_words:,} Wörter)")
        failures += not ok
    else:
        print("  [SKIP] train.txt nicht verfügbar für Statistik")

    # ── 8. Byte-Fallback-Anteil ───────────────────────────────────────────────
    print("\n── 8. Byte-Fallback-Anteil im Vokabular")
    byte_count = sum(1 for i in range(vocab_size) if sp.IsByte(i))
    byte_pct = byte_count / vocab_size * 100
    ok = check("Byte-Tokens < 5% des Vokabulars", byte_pct < 5,
               f"{byte_count} Byte-Tokens ({byte_pct:.1f}%)")
    failures += not ok

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    if failures == 0:
        print(f"✓  Alle Checks bestanden — Tokenizer sieht gut aus.")
    else:
        print(f"✗  {failures} Check(s) fehlgeschlagen — Tokenizer neu trainieren!")
        print("   Häufige Ursachen:")
        print("   - Byte-Tokens für Umlaute → character_coverage erhöhen (z.B. 0.99995)")
        print("   - Schlechte Tokens/Wort   → vocab_size anpassen")
        print("   - Suffix fehlt            → treat_whitespace_as_suffix prüfen")
    print(f"{'═'*50}")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
