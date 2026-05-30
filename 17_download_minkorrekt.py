#!/usr/bin/env python3
"""
Lädt Transkripte aller Methodisch-Inkorrekt-Episoden herunter.

Quelle:  Methodisch Inkorrekt! von Nicolas Wöhrl & Reinhard Remfort
         https://minkorrekt.de / https://minkorrekt.podigee.io
Lizenz:  CC 3.0 (Metaebene/Podigee-Feed) — BY-NC-SA anzunehmen

Strategie:
  - Neuere Episoden (Mi380+): direkter CDN-VTT-Link aus dem RSS-Feed
  - Ältere Episoden: kein Transkript verfügbar, werden übersprungen

Output: data/minkorrekt_de.txt

Usage:
  .venv_ml/bin/python 17_download_minkorrekt.py [--dry-run]
"""

import argparse
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

RSS_URL  = "https://minkorrekt.podigee.io/feed/mp3"
OUT_FILE = Path("data/minkorrekt_de.txt")


def fetch(url: str, timeout: int = 20) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Fehler {url[:60]}: {e}")
        return None


def parse_rss(feed: str) -> list[tuple[str, str | None]]:
    """Gibt Liste von (titel, vtt_url_or_None) zurück."""
    items = re.findall(r"<item>(.*?)</item>", feed, re.DOTALL)
    result = []
    for item in items:
        title = re.search(r"<title><!\[CDATA\[(.*?)\]\]>|<title>(.*?)</title>", item)
        # Beide Domains erfassen: minkorrekt.podigee.io und minkorrekt.de
        link  = re.search(r"<link>(https://minkorrekt(?:\.de|\.podigee\.io)/[^<]+)</link>", item)
        vtt   = re.search(r'podcast:transcript url="(https://[^"]+\.vtt)"', item)
        t = (title.group(1) or title.group(2) or "?").strip() if title else "?"
        v = vtt.group(1).strip() if vtt else None
        if link:
            result.append((t, v))
    return result


def parse_vtt(vtt: str) -> list[str]:
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line:
            continue
        if re.match(r"^\d+$", line) or line.startswith("NOTE"):
            continue
        line = re.sub(r"^(Speaker \d+|<v [^>]+>)[\s:]*", "", line)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Lade RSS-Feed …")
    feed = fetch(RSS_URL)
    if not feed:
        print("Feed nicht erreichbar.")
        return

    episodes = parse_rss(feed)
    print(f"{len(episodes)} Episoden im Feed\n")

    total = 0
    with (OUT_FILE.open("a", encoding="utf-8") if not args.dry_run
          else open("/dev/null", "w")) as fout:

        for i, (title, vtt_url) in enumerate(episodes):
            short = title[:50]

            if not vtt_url:
                print(f"  [{i+1:3d}/{len(episodes)}] {short} — kein Transkript")
                continue

            vtt = fetch(vtt_url)
            sentences = parse_vtt(vtt) if vtt else []

            if not sentences:
                print(f"  [{i+1:3d}/{len(episodes)}] {short} — leer")
                time.sleep(0.2)
                continue

            print(f"  [{i+1:3d}/{len(episodes)}] {short} (VTT: {len(sentences)} Sätze)")
            for s in sentences:
                fout.write(s + "\n")
            total += len(sentences)
            time.sleep(0.3)

    print(f"\nGesamt: {total} Sätze {'(dry-run)' if args.dry_run else f'-> {OUT_FILE}'}")


if __name__ == "__main__":
    main()
