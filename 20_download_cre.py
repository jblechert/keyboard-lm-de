#!/usr/bin/env python3
"""
Lädt VTT-Transkripte von CRE: Technik, Kultur, Gesellschaft herunter.

Quelle:  CRE: Technik, Kultur, Gesellschaft von Tim Pritlove (Metaebene Personal Media)
         https://cre.fm
Lizenz:  CC BY-NC-SA 3.0 DE
         https://creativecommons.org/licenses/by-nc-sa/3.0/de/
         Namensnennung: Tim Pritlove, cre.fm

URL-Muster (aus RSS-Feed):
  https://cre.fm/creNNN-slug?podlove_transcript=webvtt

Output: data/cre_de.txt

Usage:
  .venv_ml/bin/python 20_download_cre.py [--dry-run]
"""

import argparse
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

RSS_URL  = "https://cre.fm/feed/mp3"
OUT_FILE = Path("data/cre_de.txt")


def fetch(url: str, timeout: int = 20) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Fehler {url[:70]}: {e}")
        return None


def parse_rss(feed: str) -> list[tuple[str, str]]:
    """Gibt Liste von (titel, vtt_url) für CRE-Episoden zurück."""
    items = re.findall(r"<item>(.*?)</item>", feed, re.DOTALL)
    result = []
    for item in items:
        title = re.search(r"<title><!\[CDATA\[(.*?)\]\]>|<title>(.*?)</title>", item)
        link  = re.search(r"<link>(https://cre\.fm/cre\d+[^<]*)</link>", item)
        if not link:
            continue  # Sonderseiten (Hinweise etc.) überspringen
        t = (title.group(1) or title.group(2) or "?").strip() if title else "?"
        vtt_url = link.group(1).strip() + "?podlove_transcript=webvtt"
        result.append((t, vtt_url))
    return result


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
    print(f"{len(episodes)} Episoden gefunden\n")

    total = 0
    with (OUT_FILE.open("a", encoding="utf-8") if not args.dry_run
          else open("/dev/null", "w")) as fout:

        for i, (title, vtt_url) in enumerate(episodes):
            short = title[:55]
            vtt = fetch(vtt_url)
            if not vtt or not vtt.strip().startswith("WEBVTT"):
                print(f"  [{i+1:3d}/{len(episodes)}] {short} — kein Transkript")
                time.sleep(0.2)
                continue

            sentences = parse_vtt(vtt)
            if not sentences:
                print(f"  [{i+1:3d}/{len(episodes)}] {short} — leer")
                time.sleep(0.2)
                continue

            print(f"  [{i+1:3d}/{len(episodes)}] {short} ({len(sentences)} Sätze)")
            for s in sentences:
                fout.write(s + "\n")
            total += len(sentences)
            time.sleep(0.3)

    print(f"\nGesamt: {total} Sätze {'(dry-run)' if args.dry_run else f'-> {OUT_FILE}'}")


if __name__ == "__main__":
    main()
