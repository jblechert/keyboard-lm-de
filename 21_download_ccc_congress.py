#!/usr/bin/env python3
"""
Lädt deutsche VTT-Transkripte von media.ccc.de für CCC Congress-Talks.

Quelle:  Chaos Computer Club / media.ccc.de
Lizenz:  CC BY 3.0 (alle Congress-Aufzeichnungen auf media.ccc.de)
         https://creativecommons.org/licenses/by/3.0/

Abgedeckte Congresse: 32C3 bis aktuell
API:     https://api.media.ccc.de/public/

Output:  data/ccc_congress_de.txt

Usage:
  .venv_ml/bin/python 21_download_ccc_congress.py [--dry-run]
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

API_BASE = "https://api.media.ccc.de/public"
OUT_FILE = Path("data/ccc_congress_de.txt")

CONFERENCES = [
    "38c3", "37c3", "36c3", "35c3", "34c3", "33c3", "32c3",
]


def fetch_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  Fehler {url[:70]}: {e}")
        return None


def fetch_text(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Fehler {url[:70]}: {e}")
        return None


def parse_vtt(vtt: str) -> list[str]:
    lines = []
    in_note = False
    for line in vtt.splitlines():
        line = line.strip()
        if not line:
            in_note = False
            continue
        if line.startswith("NOTE"):
            in_note = True
            continue
        if in_note:
            continue
        if line.startswith("WEBVTT") or "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        line = re.sub(r"^<v [^>]+>", "", line)
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            lines.append(line)

    merged, buf = [], ""
    for line in lines:
        buf = (buf + " " + line).strip() if buf else line
        if re.search(r"[.!?]\s*$", buf):
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return [s for s in merged if len(s.split()) >= 4]


def get_vtt_url(event_url: str) -> str | None:
    """Holt die VTT-URL aus einem Event (deutsche Sprache bevorzugt)."""
    event = fetch_json(event_url)
    if not event:
        return None
    for rec_stub in event.get("recordings", []):
        if rec_stub.get("mime_type") != "text/vtt":
            continue
        rec = fetch_json(rec_stub["url"])
        if rec and rec.get("language") == "deu":
            return rec.get("recording_url")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = 0
    with (OUT_FILE.open("a", encoding="utf-8") if not args.dry_run
          else open("/dev/null", "w")) as fout:

        for conf in CONFERENCES:
            print(f"\n── {conf.upper()} ──")
            data = fetch_json(f"{API_BASE}/conferences/{conf}")
            if not data:
                continue

            events = [e for e in data.get("events", [])
                      if e.get("original_language") == "deu"]
            print(f"{len(events)} deutsche Talks")

            for i, event in enumerate(events):
                title = event.get("title", "?")[:50]
                vtt_url = get_vtt_url(event["url"])

                if not vtt_url:
                    print(f"  [{i+1:3d}/{len(events)}] {title} — kein VTT")
                    time.sleep(0.1)
                    continue

                vtt = fetch_text(vtt_url)
                sentences = parse_vtt(vtt) if vtt else []

                if not sentences:
                    print(f"  [{i+1:3d}/{len(events)}] {title} — leer")
                    time.sleep(0.2)
                    continue

                print(f"  [{i+1:3d}/{len(events)}] {title} ({len(sentences)} Sätze)")
                for s in sentences:
                    fout.write(s + "\n")
                total += len(sentences)
                time.sleep(0.4)

    print(f"\nGesamt: {total} Sätze {'(dry-run)' if args.dry_run else f'-> {OUT_FILE}'}")


if __name__ == "__main__":
    main()
