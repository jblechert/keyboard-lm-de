#!/usr/bin/env python3
"""
Generates high-quality German keyboard training sentences via a local LLM
(Ollama OpenAI-compatible API, e.g. Qwen3.6-27b).

The goal is a small but high-quality set that represents how people
actually type on a phone — short, natural, varied German.

Output: data/synthetic_de.txt  (one sentence per line)

Usage:
  .venv_ml/bin/python 08_generate_synthetic.py [--target 2000] [--model qwen3:27b]
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Fehler: openai-Paket fehlt. Installieren mit:", file=sys.stderr)
    print("  .venv_ml/bin/pip install openai", file=sys.stderr)
    sys.exit(1)

OUT = Path("data/synthetic_de.txt")

# ── Themen & Stil-Variationen ──────────────────────────────────────────────────
# Jeder Eintrag: (thema, konkrete_situation)
TOPICS = [
    ("Verabredungen",   "Du machst Pläne mit einer Freundin — Uhrzeit, Ort, wer kommt"),
    ("Essen & Hunger",  "Du schreibst jemandem was du essen willst, gerade isst oder gekocht hast"),
    ("Arbeit",          "Kurze Nachrichten zwischen Kollegen — Meeting, Aufgabe, Feierabend"),
    ("Familie",         "Du schreibst deiner Mutter, deinem Partner oder deinen Kindern"),
    ("Einkaufen",       "Einkaufsliste, was fehlt, ob jemand etwas mitbringen soll"),
    ("Wetter",          "Jemand kommentiert das Wetter oder fragt danach"),
    ("Zustimmung/Absage", "Kurze Zu- oder Absagen, Bestätigungen, Absicherungen"),
    ("Unterwegs",       "Jemand ist in der Bahn, im Auto, zu Fuß — gibt Bescheid"),
    ("Freizeit",        "Pläne für Wochenende, Sport, Film, Konzert, Spaziergang"),
    ("Kleine Updates",  "Kurze Statusmeldungen — 'Bin gleich da', 'Hat geklappt', 'Alles gut'"),
    ("Fragen stellen",  "Alltägliche Fragen die man per Handy stellt"),
    ("Meinungen",       "Jemand äußert kurz seine Meinung zu etwas Alltäglichem"),
    ("Glückwünsche",    "Geburtstag, Prüfung, neue Stelle, Geburt — kurze herzliche Nachrichten"),
    ("Entschuldigungen","Sich kurz entschuldigen, verspätet sein, etwas vergessen haben"),
    ("Technik/Apps",    "Kurze Sätze rund um Handy, App, Internet, Akku"),
]

STYLES = [
    "sehr kurz und knapp (3–6 Wörter), so wie man tippt wenn man es eilig hat",
    "informell, du-Form, locker wie unter Freunden",
    "freundlich aber direkt, keine Füllwörter",
    "als Frage formuliert",
    "mit einem Ausrufezeichen oder Emoji am Ende (1–2 Sätze mit Emoji, Rest ohne)",
    "etwas länger (10–15 Wörter), aber immer noch natürlich und flüssig",
]

SYSTEM_PROMPT = """Du generierst deutsche Sätze so wie echte Menschen sie auf dem Smartphone tippen.

Wichtige Regeln:
- Natürlich und gesprochen, KEIN Bürokratendeutsch, kein Wikipedia-Stil
- Kurz: meist 4–12 Wörter, gelegentlich kürzer oder etwas länger
- Korrekte Rechtschreibung und Grammatik (Tippfehler weglassen)
- Keine Anführungszeichen, keine Nummerierung, kein Kommentar
- Genau eine Zeile pro Satz, keine Leerzeilen zwischen Sätzen
- Abwechslungsreich: kein Satz darf dem vorherigen ähneln"""


def build_user_prompt(topic: str, situation: str, style: str, n: int) -> str:
    # /no_think muss bei Qwen3 am Anfang der User-Message stehen
    return (
        f"/no_think\n"
        f"Thema: {topic}\n"
        f"Situation: {situation}\n"
        f"Stil: {style}\n\n"
        f"Schreib genau {n} solche Sätze, einen pro Zeile."
    )


def parse_sentences(raw: str) -> list[str]:
    """Extract clean sentences from model output."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Remove numbering like "1.", "1)", "-", "•"
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-•]\s*', '', line)
        # Skip empty lines, meta-comments, or lines that look like headers
        if not line:
            continue
        if line.endswith(':') or len(line) < 5:
            continue
        # Skip lines with quotes around the whole thing
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000,
                        help="Anzahl Sätze (default: 2000)")
    parser.add_argument("--model", default="qwen3:27b",
                        help="Ollama-Modellname (default: qwen3:27b)")
    parser.add_argument("--host", default="http://localhost:11434/v1",
                        help="Ollama API base URL")
    parser.add_argument("--batch", type=int, default=20,
                        help="Sätze pro API-Aufruf (default: 20)")
    parser.add_argument("--append", action="store_true",
                        help="An bestehende Datei anhängen statt überschreiben")
    args = parser.parse_args()

    client = OpenAI(base_url=args.host, api_key="ollama")

    mode = "a" if args.append else "w"
    existing = 0
    if args.append and OUT.exists():
        existing = sum(1 for _ in OUT.open())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    errors = 0
    t_start = time.time()

    print(f"Ziel: {args.target} Sätze → {OUT}")
    print(f"Modell: {args.model}  |  Batch: {args.batch} Sätze/Aufruf")
    if existing:
        print(f"Vorhandene Sätze: {existing} (wird angehängt)")
    print()

    with OUT.open(mode, encoding="utf-8") as f:
        while collected < args.target:
            topic, situation = random.choice(TOPICS)
            style = random.choice(STYLES)
            remaining = args.target - collected
            n = min(args.batch, remaining)

            user_msg = build_user_prompt(topic, situation, style, n)

            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.9,
                    max_tokens=n * 30,
                )
                raw = resp.choices[0].message.content or ""
                sentences = parse_sentences(raw)

                if not sentences:
                    errors += 1
                    if errors > 5:
                        print("Zu viele leere Antworten — Modell läuft?", file=sys.stderr)
                        sys.exit(1)
                    continue

                for s in sentences:
                    f.write(s + "\n")
                collected += len(sentences)

                elapsed = time.time() - t_start
                rate = collected / elapsed if elapsed > 0 else 0
                eta = (args.target - collected) / rate if rate > 0 else 0
                print(f"  [{collected:>5}/{args.target}]  "
                      f"Thema: {topic:<20}  "
                      f"{rate:.0f} Sätze/s  "
                      f"ETA: {eta/60:.1f} min")

            except KeyboardInterrupt:
                print(f"\nAbgebrochen. {collected} Sätze gespeichert.")
                break
            except Exception as e:
                print(f"  Fehler: {e}", file=sys.stderr)
                errors += 1
                time.sleep(2)

    total = time.time() - t_start
    print(f"\nFertig: {collected} Sätze in {total/60:.1f} min → {OUT}")
    if collected > 0:
        print(f"Durchschnitt: {total/collected:.2f}s pro Satz")


if __name__ == "__main__":
    main()
