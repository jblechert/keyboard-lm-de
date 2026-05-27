#!/usr/bin/env python3
"""
Vergleicht alle GGUF-Varianten einer Modellversion.

Für jede GGUF-Datei werden gemessen:
  - Dateigröße (MB)
  - Prompt-Processing-Geschwindigkeit (pp, tokens/s) — wie schnell Kontext verarbeitet wird
  - Token-Generierungsgeschwindigkeit (tg, tokens/s) — wie schnell Vorhersagen kommen
  - Relative Geschwindigkeit gegenüber F16

Gibt eine Markdown-Tabelle aus, direkt für Release-Notes verwendbar.

Usage:
  .venv_ml/bin/python 13_compare_gguf.py --pattern "data/mjb-de-200k*.gguf"
  .venv_ml/bin/python 13_compare_gguf.py data/mjb-de-200k-F16.gguf data/mjb-de-200k-Q4_0.gguf
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Keyboard-LM-relevante Benchmark-Parameter:
# - kurzer Prompt (64 Token) simuliert typischen Tastatur-Kontext
# - 1 generierter Token (nur eine Vorhersage nötig)
BENCH_PROMPT_TOKENS = 64
BENCH_GEN_TOKENS    = 1
BENCH_REPETITIONS   = 3

# Bekannte Quantisierungen in sinnvoller Reihenfolge
QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q4_0", "Q3_K", "Q2_K"]


def find_ggufs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if "*" in pat or "?" in pat:
            files.extend(sorted(p.parent.glob(p.name)))
        elif p.exists():
            files.append(p)
        else:
            print(f"Warnung: {pat} nicht gefunden", file=sys.stderr)
    # deduplizieren, Reihenfolge beibehalten
    seen: set[Path] = set()
    result = []
    for f in files:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def quant_key(path: Path) -> int:
    name = path.stem.upper()
    for i, q in enumerate(QUANT_ORDER):
        if q in name:
            return i
    return len(QUANT_ORDER)


def detect_quant(path: Path) -> str:
    name = path.stem.upper()
    for q in QUANT_ORDER:
        if q in name:
            return q
    return "?"


def run_bench(gguf: Path, repetitions: int) -> tuple[float, float] | None:
    """
    Gibt (pp_tok_per_s, tg_tok_per_s) zurück oder None bei Fehler.
    pp = prompt processing, tg = token generation
    """
    cmd = [
        "llama-bench",
        "-m", str(gguf),
        "-p", str(BENCH_PROMPT_TOKENS),
        "-n", str(BENCH_GEN_TOKENS),
        "-r", str(repetitions),
        "-o", "json",
        "--no-warmup",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {gguf.name}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Fehler: llama-bench nicht gefunden. Installiere llama.cpp.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"  llama-bench Fehler: {result.stderr.strip()}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  Ungültige Ausgabe von llama-bench: {result.stdout[:200]}", file=sys.stderr)
        return None

    pp_speed = tg_speed = None
    for entry in data:
        n_prompt = entry.get("n_prompt", 0)
        n_gen    = entry.get("n_gen", 0)
        speed    = entry.get("avg_ts", 0)
        if n_prompt > 0 and n_gen == 0:
            pp_speed = speed
        elif n_gen > 0 and n_prompt == 0:
            tg_speed = speed

    # llama-bench gibt manchmal pp+tg kombiniert aus
    if pp_speed is None and tg_speed is None and data:
        for entry in data:
            t = entry.get("test", "")
            speed = entry.get("avg_ts", 0)
            if "pp" in t:
                pp_speed = speed
            elif "tg" in t:
                tg_speed = speed

    return (pp_speed, tg_speed) if (pp_speed or tg_speed) else None


def format_speed(speed: float | None, ref: float | None) -> tuple[str, str]:
    if speed is None:
        return "—", "—"
    s = f"{speed:,.0f}"
    if ref and ref > 0:
        rel = speed / ref * 100
        return s, f"{rel:.0f}%"
    return s, "100%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*",
                        help="GGUF-Dateien oder Glob-Pattern")
    parser.add_argument("--pattern", "-p",
                        help="Glob-Pattern für GGUF-Dateien (z.B. 'data/mjb-de-200k*.gguf')")
    parser.add_argument("--repetitions", "-r", type=int, default=BENCH_REPETITIONS,
                        help=f"Wiederholungen pro Modell (default: {BENCH_REPETITIONS})")
    parser.add_argument("--no-bench", action="store_true",
                        help="Nur Dateigrößen, kein Benchmark")
    parser.add_argument("--markdown", action="store_true", default=True,
                        help="Markdown-Ausgabe (default: an)")
    args = parser.parse_args()

    patterns = list(args.files)
    if args.pattern:
        patterns.append(args.pattern)
    if not patterns:
        patterns = ["data/*.gguf"]

    ggufs = sorted(find_ggufs(patterns), key=quant_key)
    if not ggufs:
        print("Keine GGUF-Dateien gefunden.", file=sys.stderr)
        sys.exit(1)

    print(f"Gefunden: {len(ggufs)} GGUF-Dateien\n", file=sys.stderr)

    # Messwerte sammeln
    rows = []
    for gguf in ggufs:
        quant    = detect_quant(gguf)
        size_mb  = gguf.stat().st_size / 1024 / 1024
        print(f"  Benchmark: {gguf.name} …", end=" ", flush=True, file=sys.stderr)

        if args.no_bench:
            bench = None
            print("übersprungen", file=sys.stderr)
        else:
            bench = run_bench(gguf, args.repetitions)
            if bench:
                print(f"pp={bench[0]:,.0f} t/s  tg={bench[1]:,.0f} t/s", file=sys.stderr)
            else:
                print("Fehler", file=sys.stderr)

        rows.append({
            "name":    gguf.name,
            "quant":   quant,
            "size_mb": size_mb,
            "pp":      bench[0] if bench else None,
            "tg":      bench[1] if bench else None,
        })

    # F16 als Referenz
    ref_pp = next((r["pp"] for r in rows if r["quant"] == "F16" and r["pp"]), None)
    ref_tg = next((r["tg"] for r in rows if r["quant"] == "F16" and r["tg"]), None)

    # Markdown-Tabelle ausgeben
    print()
    if args.no_bench:
        print(f"| Variante | Größe |")
        print(f"|---|---|")
        for r in rows:
            print(f"| `{r['quant']}` | {r['size_mb']:.0f} MB |")
    else:
        print(f"| Variante | Größe | Prompt (t/s) | Vorhersage (t/s) | Rel. Geschw. |")
        print(f"|---|---|---|---|---|")
        for r in rows:
            pp_s, pp_rel = format_speed(r["pp"], ref_pp)
            tg_s, _      = format_speed(r["tg"], ref_tg)
            # Relative Geschwindigkeit: nutze tg falls pp fehlt
            _, rel       = format_speed(r["tg"] or r["pp"],
                                        ref_tg or ref_pp)
            print(f"| `{r['quant']}` | {r['size_mb']:.0f} MB "
                  f"| {pp_s} | {tg_s} | {rel} |")

    print()
    print(f"*Benchmark: {BENCH_PROMPT_TOKENS}-Token-Kontext, "
          f"{BENCH_GEN_TOKENS} generierter Token, "
          f"{args.repetitions}× wiederholt*")


if __name__ == "__main__":
    main()
