#!/usr/bin/env python3
"""
Lädt VTT-Transkripte von Raumzeit herunter und extrahiert saubere deutsche Sätze.

Quelle:  Raumzeit von Tim Pritlove (Metaebene Personal Media)
         https://raumzeit-podcast.de
Lizenz:  CC BY-NC-SA 3.0 DE
         https://creativecommons.org/licenses/by-nc-sa/3.0/de/
         Namensnennung: Tim Pritlove, raumzeit-podcast.de

URL-Muster:
  https://raumzeit-podcast.de/rzNNN?podlove_transcript=webvtt

Output: data/raumzeit_de.txt

Usage:
  .venv_ml/bin/python 18_download_raumzeit.py [--dry-run] [--start 1] [--end 200]
"""

import argparse
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://raumzeit-podcast.de/rz{:03d}?podlove_transcript=webvtt"
OUT_FILE  = Path("data/raumzeit_de.txt")


def fetch_vtt(n: int) -> str | None:
    url = BASE_URL.format(n)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            if content.strip().startswith("WEBVTT"):
                return content
            return None
    except urllib.error.HTTPError as e:
        if e.code in (404, 301):
            return None
        print(f"  HTTP {e.code} für RZ{n:03d}")
        return None
    except Exception as e:
        print(f"  Fehler bei RZ{n:03d}: {e}")
        return None


def parse_vtt(vtt: str) -> list[str]:
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE") or "-->" in line:
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
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=300)
    args = parser.parse_args()

    total = 0
    consecutive_missing = 0

    with (OUT_FILE.open("a", encoding="utf-8") if not args.dry_run
          else open("/dev/null", "w")) as fout:

        for n in range(args.start, args.end + 1):
            vtt = fetch_vtt(n)

            if vtt is None:
                consecutive_missing += 1
                if consecutive_missing >= 5:
                    print(f"  5 aufeinanderfolgende Fehler — stoppe bei RZ{n:03d}")
                    break
                continue

            consecutive_missing = 0
            sentences = parse_vtt(vtt)
            print(f"  RZ{n:03d}: {len(sentences)} Sätze")

            for s in sentences:
                fout.write(s + "\n")
            total += len(sentences)
            time.sleep(0.3)

    print(f"\nGesamt: {total} Sätze {'(dry-run)' if args.dry_run else f'-> {OUT_FILE}'}")


if __name__ == "__main__":
    main()
