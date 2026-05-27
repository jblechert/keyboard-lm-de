#!/usr/bin/env python3
"""
Trains a SentencePiece tokenizer on all available training data and saves it to
data/tokenizer/de_keyboard.model + .vocab.

Key settings that match FUTO's format:
  - treat_whitespace_as_suffix=true  → spaces at end of tokens ("wort▁")
  - vocab_size=15008                 → matches FUTO English model
  - model_type=bpe                   → Byte Pair Encoding
  - Special tokens for autocorrect   → <XBU> <XBC> <XEC> <CHAR_A>…<CHAR_Z>
"""

import sys
from pathlib import Path
import sentencepiece as spm

# All sources — existing files are used automatically
SOURCES = [
    Path("data/tatoeba_de.txt"),
    Path("data/c4_de.txt"),
    Path("data/synthetic_de.txt"),
    Path("data/synthetic_themen.txt"),
]
OUT_DIR     = Path("data/tokenizer")
MODEL_NAME  = "de_keyboard"

VOCAB_SIZE  = 15_008

# FUTO special tokens (added on top of the learned vocabulary)
FUTO_SPECIAL_TOKENS = [
    "<XBU>",   # begin user input (start of a misspelled word, char by char)
    "<XBC>",   # begin correction
    "<XEC>",   # end correction
    *[f"<CHAR_{chr(c)}>" for c in range(ord("A"), ord("Z") + 1)],  # <CHAR_A>…<CHAR_Z>
]


def main():
    inputs = [p for p in SOURCES if p.exists()]
    if not inputs:
        print("Fehler: Keine Trainingsdaten gefunden.", file=sys.stderr)
        print("  Erst 07_download_tatoeba.py und/oder 09_download_c4_de.py ausführen.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_special = len(FUTO_SPECIAL_TOKENS)
    vocab_for_sp = VOCAB_SIZE - n_special

    input_str = ",".join(str(p) for p in inputs)
    total_mb = sum(p.stat().st_size for p in inputs) / 1024 / 1024
    print(f"Trainiere SentencePiece-Tokenizer auf {len(inputs)} Quelle(n), {total_mb:.0f} MB:")
    for p in inputs:
        print(f"  {p}  ({p.stat().st_size/1024/1024:.0f} MB)")
    print(f"  Vokabular: {vocab_for_sp} gelernt + {n_special} Special Tokens = {VOCAB_SIZE} gesamt")

    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=str(OUT_DIR / MODEL_NAME),
        vocab_size=vocab_for_sp,
        model_type="bpe",

        # FUTO-spezifisch: Leerzeichen am Ende, nicht am Anfang
        treat_whitespace_as_suffix=True,

        # Zeichenabdeckung: 0.9999 deckt praktisch alle deutschen Zeichen ab
        character_coverage=0.9999,

        # Sonderzeichen-Handling
        byte_fallback=True,          # unbekannte Bytes als <0xNN>-Token
        add_dummy_prefix=False,      # kein Dummy-Leerzeichen am Satzanfang

        # SP sampelt zufällig — 2M Sätze reichen für gutes Vokabular
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,

        # Reservierte Plätze für unsere Special Tokens
        user_defined_symbols=FUTO_SPECIAL_TOKENS,

        # Standard-Kontrolltoken
        pad_id=3,
    )

    model_path = OUT_DIR / f"{MODEL_NAME}.model"
    vocab_path = OUT_DIR / f"{MODEL_NAME}.vocab"
    print(f"\nFertig:")
    print(f"  Modell: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Vokab:  {vocab_path}")

    # Kurzer Selbsttest
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    testfälle = [
        "Das Modell lernt deutsche Sprache.",
        "schreiben schreibt schrieb",
        "Österreich Überzeugung Ärger",
    ]
    print("\nSelbsttest:")
    for text in testfälle:
        tokens = sp.encode(text, out_type=str)
        print(f"  {text!r}")
        print(f"    → {tokens}")


if __name__ == "__main__":
    main()
