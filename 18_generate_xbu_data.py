#!/usr/bin/env python3
"""
Converts plain German training sentences into XBU/CHAR autocorrect format
for LoRA fine-tuning.

Format: [context] <XBU><CHAR_X><CHAR_Y>...<XBC>word <XEC>

Mirrors FUTO Keyboard's TrainingDataGenerator.kt logic exactly:
- QWERTY key positions with Gaussian tap noise (temperature=1.0)
- Random transpose, adjacent-transpose, delete mutations
- Random word trimming (partial words, as if user hasn't finished typing)
- ~1/3 of words per sentence converted to XBU format

Usage:
  .venv_ml/bin/python 18_generate_xbu_data.py
  .venv_ml/bin/python 18_generate_xbu_data.py --proportion 0.33 --correctness 0.8
  .venv_ml/bin/python 18_generate_xbu_data.py --input data/tatoeba_de.txt --output data/xbu_tatoeba.txt
"""

import argparse
import math
import random
import re
import sys
from pathlib import Path

# ── QWERTY layout (pixel positions from TrainingDataGenerator.kt) ─────────────

TAP_SIZE = (80.0, 80.0)

KEYBOARD_KEYS = {
    'q': (75.0,   106.0), 'w': (214.0,  106.0), 'e': (363.0,  106.0),
    'r': (499.0,  106.0), 't': (645.0,  106.0), 'y': (789.0,  106.0),
    'u': (928.0,  106.0), 'i': (1073.0, 106.0), 'o': (1216.0, 106.0),
    'p': (1357.0, 106.0),
    'a': (150.0,  312.0), 's': (291.0,  312.0), 'd': (434.0,  312.0),
    'f': (574.0,  312.0), 'g': (717.0,  312.0), 'h': (859.0,  312.0),
    'j': (1005.0, 312.0), 'k': (1140.0, 312.0), 'l': (1288.0, 312.0),
    'z': (287.0,  515.0), 'x': (434.0,  515.0), 'c': (576.0,  515.0),
    'v': (718.0,  515.0), 'b': (860.0,  515.0), 'n': (1003.0, 515.0),
    'm': (1145.0, 515.0),
}

def _closest_key(x: float, y: float) -> str:
    return min(KEYBOARD_KEYS, key=lambda c: (KEYBOARD_KEYS[c][0]-x)**2 + (KEYBOARD_KEYS[c][1]-y)**2)

def _random_normal(mean: float, std: float) -> float:
    return random.gauss(mean, std)

def substitute_keyboard_letters(word: str, temperature: float = 0.6) -> str:
    result = []
    for ch in word.lower():
        if ch not in KEYBOARD_KEYS:
            continue
        kx, ky = KEYBOARD_KEYS[ch]
        nx = _random_normal(kx, temperature * TAP_SIZE[0])
        ny = _random_normal(ky, temperature * TAP_SIZE[1])
        result.append(_closest_key(nx, ny))
    return ''.join(result)

def transpose_random_letters(word: str) -> str:
    if len(word) < 2:
        return word
    a = random.randrange(len(word))
    b = random.randrange(len(word))
    while b == a:
        b = random.randrange(len(word))
    lst = list(word)
    lst[a], lst[b] = lst[b], lst[a]
    return ''.join(lst)

def transpose_adjacent_letters(word: str) -> str:
    if len(word) < 2:
        return word
    i = random.randrange(len(word) - 1)
    lst = list(word)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return ''.join(lst)

def delete_random_character(word: str) -> str:
    if not word:
        return word
    i = random.randrange(len(word))
    return word[:i] + word[i+1:]

def misspell_word(word: str, correctness: float = 0.8) -> str:
    """Mirrors WordMisspelling.misspellWord() from TrainingDataGenerator.kt."""
    misspelled = word.strip().lower().replace("'", "")

    def get_rand():
        return random.random() ** correctness

    if get_rand() > 0.5:
        misspelled = transpose_random_letters(misspelled)
    if get_rand() > 0.5:
        misspelled = transpose_adjacent_letters(misspelled)
    if get_rand() > 0.5:
        misspelled = delete_random_character(misspelled)

    misspelled = substitute_keyboard_letters(misspelled, temperature=1.0 * get_rand())

    # Trim to partial word (user hasn't finished typing)
    if get_rand() > 0.33 and len(misspelled) >= 2:
        new_len = math.ceil((1.0 - (get_rand() ** 2)) * len(misspelled))
        new_len = max(2, min(new_len, len(misspelled)))
        misspelled = misspelled[:new_len]

    return misspelled

# ── XBU token format ──────────────────────────────────────────────────────────

PERMITTED = set('abcdefghijklmnopqrstuvwxyz')

def format_word_misspelling(misspelled: str, truth: str) -> str:
    """<XBU><CHAR_X>...<XBC>word <XEC>"""
    chars = [c for c in misspelled if c in PERMITTED]
    if not chars:
        return ''
    char_tokens = ''.join(f'<CHAR_{c.upper()}>' for c in chars)
    return f'<XBU>{char_tokens}<XBC>{truth.strip()} <XEC>'

def convert_sentence(sentence: str, proportion: float = 0.33, correctness: float = 0.8) -> str:
    """
    Takes a plain sentence and randomly converts ~proportion of words to XBU format.
    Returns the mixed sentence, or empty string if nothing could be converted.
    """
    words = sentence.strip().split()
    if not words:
        return ''

    n_to_convert = max(1, round(len(words) * proportion))
    # Only convert words that consist of plain ASCII letters (no German special chars —
    # CHAR tokens only cover a-z, matching FUTO's permittedCharacters)
    convertible = [i for i, w in enumerate(words)
                   if w.lower().replace("'", "").isalpha()
                   and all(c in PERMITTED for c in w.lower().replace("'", ""))
                   and len(w) >= 2]

    if not convertible:
        return sentence  # return as-is (pure LM training line)

    to_convert = random.sample(convertible, min(n_to_convert, len(convertible)))

    result = list(words)
    for i in to_convert:
        truth = words[i]
        misspelled = misspell_word(truth, correctness)
        formatted = format_word_misspelling(misspelled, truth)
        if formatted:
            result[i] = formatted

    return ' '.join(result)


# ── Sources ───────────────────────────────────────────────────────────────────

DEFAULT_SOURCES = [
    Path("data/tatoeba_de.txt"),
    *sorted(Path("data").glob("synthetic_*.txt")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",       nargs="*", metavar="FILE",
                    help="Input files (default: tatoeba + synthetic_*.txt)")
    ap.add_argument("--output",      default="data/xbu_lora.txt",
                    help="Output file (default: data/xbu_lora.txt)")
    ap.add_argument("--proportion",  type=float, default=0.33,
                    help="Fraction of words per sentence to XBU-convert (default: 0.33)")
    ap.add_argument("--correctness", type=float, default=0.8,
                    help="Typing correctness 0–1, lower = more typos (default: 0.8)")
    ap.add_argument("--copies",      type=int, default=3,
                    help="How many XBU-augmented copies per sentence (default: 3)")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    sources = [Path(f) for f in args.input] if args.input else DEFAULT_SOURCES
    sources = [s for s in sources if s.exists()]
    if not sources:
        print("Fehler: Keine Quelldateien gefunden.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_in = total_out = 0

    with out.open('w', encoding='utf-8') as fout:
        for src in sources:
            lines = src.read_text(encoding='utf-8').splitlines()
            print(f"  {src.name}: {len(lines):,} Sätze")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                total_in += 1
                # Always write original (plain LM training)
                fout.write(line + '\n')
                total_out += 1
                # Write N XBU-augmented copies
                for _ in range(args.copies):
                    converted = convert_sentence(line, args.proportion, args.correctness)
                    if converted and converted != line:
                        fout.write(converted + '\n')
                        total_out += 1

    print(f"\nFertig: {total_in:,} Eingabe-Sätze → {total_out:,} Ausgabe-Zeilen")
    print(f"  (1 Original + bis zu {args.copies} XBU-Kopien pro Satz)")
    print(f"  → {out}  ({out.stat().st_size // 1024 // 1024} MB)")


if __name__ == '__main__':
    main()
