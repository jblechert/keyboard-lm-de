#!/usr/bin/env python3
"""
L\xe4dt VTT-Transkripte von Logbuch:Netzpolitik herunter und extrahiert
saubere deutsche S\xe4tze daraus.

Quelle:  Logbuch:Netzpolitik von Tim Pritlove & Linus Neumann (Metaebene Personal Media)
         https://logbuch-netzpolitik.de
Lizenz:  Bitte vor Verwendung pr\xfcfen — Metaebene-Podcasts nutzen \xfcblicherweise CC BY.

URL-Muster:
  https://logbuch-netzpolitik.de/lnpNNN?podlove_transcript=webvtt
  Episoden 006–554 haben Transkripte (Whisper-generiert ab ca. EP006).
  Episoden 001–005 \xfcberspringen (Qualit\xe4t unzureichend).

Output: data/lnp_de.txt

Usage:
  .venv_ml/bin/python 16_download_lnp.py [--dry-run] [--start 6] [--end 554]
"""

import argparse
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://logbuch-netzpolitik.de/lnp{:03d}?podlove_transcript=webvtt"
OUT_FILE  = Path("data/lnp_de.txt")


def fetch_vtt(n: int) -> str | None:
    url = BASE_URL.format(n)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            if content.strip().startswith("WEBVTT"):
                return content
            return None   # HTML zur\xfcckbekommen = kein Transkript
    except urllib.error.HTTPError as e:
        if e.code in (404, 301):
            return None
        print(f"  HTTP {e.code} f\xfcr LNP{n:03d}")
        return None
    except Exception as e:
        print(f"  Fehler bei LNP{n:03d}: {e}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=6,
                        help="Erste Episode (default: 6)")
    parser.add_argument("--end", type=int, default=600,
                        help="Letzte Episode (default: 600)")
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
                    print(f"  5 aufeinanderfolgende Fehler — stoppe bei LNP{n:03d}")
                    break
                continue

            consecutive_missing = 0
            sentences = parse_vtt(vtt)
            print(f"  LNP{n:03d}: {len(sentences)} S\xe4tze")

            for s in sentences:
                fout.write(s + "\n")
            total += len(sentences)
            time.sleep(0.3)

    print(f"\nGesamt: {total} S\xe4tze {'(dry-run)' if args.dry_run else f'-> {OUT_FILE}'}")


if __name__ == "__main__":
    main()
