#!/usr/bin/env python3
"""
Generiert ein sauberes Eval-Set aus frisch synthetisierten deutschen Sätzen.
Sätze sind nicht Teil des Trainings-Corpus — geeignet für fairen Modellvergleich.

Requires Ollama with qwen3.6:27b running locally.

Usage:
  .venv_ml/bin/python 13_generate_eval_set.py [--per-topic 50] [--out data/eval_clean.txt]
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

TOPICS = [
    ("Nachrichten", "lokale Ereignisse, Politik, Gesellschaft"),
    ("Kochen",      "Kochen, Rezepte, Lebensmittel, Küche"),
    ("Gartenarbeit","Garten, Pflanzen, Gartenarbeit, Natur"),
    ("Handwerk",    "Heimwerken, Reparaturen, Werkzeug, Basteln"),
    ("Gesundheit",  "Gesundheit, Arztbesuch, Sport, Wohlbefinden"),
    ("Reisen",      "Reisen, Urlaub, Orte, Unterkünfte, Ausflüge"),
    ("Familie",     "Familienalltag, Kinder, Verwandte, Zuhause"),
    ("Arbeit",      "Berufsalltag, Büro, Kollegen, Meetings, Karriere"),
    ("Sport",       "Sport, Training, Fußball, Fitness, Wettkampf"),
    ("Natur",       "Wetter, Tiere, Landschaft, Jahreszeiten"),
]

HOST  = "http://localhost:11434"
MODEL = "qwen3.6:27b"


def ask_batch(topic: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "think": False,
        "stream": False,
        "options": {"num_predict": 400, "temperature": 0.9},
        "messages": [
            {"role": "system", "content":
                "Du generierst natürliche deutsche Alltagssätze für ein Keyboard-Sprachmodell. "
                "Gib ausschließlich die Sätze aus, einen pro Zeile, ohne Nummerierung, "
                "ohne Erklärungen, ohne Anführungszeichen."},
            {"role": "user", "content":
                f"Schreibe 10 verschiedene natürliche deutsche Sätze zum Thema: {topic}"},
        ],
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-topic", type=int, default=50)
    parser.add_argument("--out", default="data/eval_clean.txt")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as f:
        for topic_name, topic_desc in TOPICS:
            print(f"  {topic_name}...", flush=True)
            count = 0
            while count < args.per_topic:
                try:
                    raw = ask_batch(topic_desc)
                    for line in raw.splitlines():
                        line = line.strip().lstrip("0123456789.-•) ").strip("\"'")
                        if len(line) > 20 and line[0].isupper() and count < args.per_topic:
                            f.write(line + "\n")
                            f.flush()
                            count += 1
                            total += 1
                except Exception as e:
                    print(f"    Fehler: {e}", flush=True)
            print(f"    {count} Sätze ({total} gesamt)", flush=True)

    print(f"Fertig: {total} Sätze → {out}")


if __name__ == "__main__":
    main()
