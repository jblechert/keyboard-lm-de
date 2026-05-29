#!/usr/bin/env python3
"""
L\xe4dt VTT-Transkripte von parlamentsrevue.de und landtagsrevue (ltr) herunter
und extrahiert saubere deutsche S\xe4tze daraus.

Quelle:  Parlamentsrevue / Landtagsrevue von Sabrina Gehder
Lizenz:  Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
         https://creativecommons.org/licenses/by-sa/4.0/
         Namensnennung: Sabrina Gehder, parlamentsrevue.de

URL-Muster:
  https://parlamentsrevue.de/wp-content/upload/PRXXX.vtt   (Parlamentsrevue, ab PR022)
  https://parlamentsrevue.de/wp-content/upload/LTRXXX.vtt  (Landtagsrevue, ab LTR001)

Output: data/parlamentsrevue_de.txt

Usage:
  .venv_ml/bin/python 15_download_parlamentsrevue.py [--dry-run] [--max-ep 100]
"""

import argparse
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://parlamentsrevue.de/wp-content/upload/{}.vtt"
OUT_FILE  = Path("data/parlamentsrevue_de.txt")

PREFIXES = [
    ("PR",  range(22, 200)),  # Parlamentsrevue (startet bei PR022)
    ("LTR", range(1, 200)),   # Landtagsrevue
]


def fetch_vtt(episode_id: str) -> str | None:
    url = BASE_URL.format(episode_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"  HTTP {e.code} f\xfcr {episode_id}")
        return None
    except Exception as e:
        print(f"  Fehler bei {episode_id}: {e}")
        return None


def parse_vtt(vtt: str) -> list[str]:
    """Extrahiert S\xe4tze aus VTT — merged aufgeteilte Segmente zu vollst\xe4ndigen S\xe4tzen."""
    # Entferne Header, Timestamps, Speaker-Tags
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line:
            continue
        if re.match(r"^\d+$", line):   # Sequenznummer
            continue
        # Speaker-Tag entfernen: <v Name>Text
        line = re.sub(r"^<v [^>]+>", "", line)
        line = re.sub(r"<[^>]+>", "", line)   # alle anderen Tags
        line = line.strip()
        if line:
            lines.append(line)

    # Segmente zusammenf\xfchren: wenn eine Zeile nicht mit Satzzeichen endet,
    # geh\xf6rt sie zur n\xe4chsten Zeile
    merged = []
    buf = ""
    for line in lines:
        if buf:
            buf += " " + line
        else:
            buf = line
        # Satzende: endet mit . ! ? oder ist lang genug und n\xe4chste beginnt gro\xdf
        if re.search(r"[.!?]\s*$", buf):
            merged.append(buf.strip())
            buf = ""

    if buf:
        merged.append(buf.strip())

    # Filter: Mindestl\xe4nge 4 W\xf6rter
    return [s for s in merged if len(s.split()) >= 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur ausgeben, nichts schreiben")
    parser.add_argument("--max-ep", type=int, default=200,
                        help="Maximale Episodennummer (default: 200)")
    args = parser.parse_args()

    total = 0
    with (OUT_FILE.open("a", encoding="utf-8") if not args.dry_run
          else open("/dev/null", "w")) as fout:

        for prefix, ep_range in PREFIXES:
            consecutive_misses = 0
            for n in ep_range:
                if n > args.max_ep:
                    break
                episode_id = f"{prefix}{n:03d}"
                vtt = fetch_vtt(episode_id)

                if vtt is None:
                    consecutive_misses += 1
                    if consecutive_misses >= 5:
                        print(f"  5 aufeinanderfolgende 404 — {prefix} fertig bei EP {n-1}")
                        break
                    continue

                consecutive_misses = 0
                sentences = parse_vtt(vtt)
                print(f"  {episode_id}: {len(sentences)} S\xe4tze")

                for s in sentences:
                    fout.write(s + "\n")
                total += len(sentences)
                time.sleep(0.5)   # h\xf6fliche Pause

    print(f"\nGesamt: {total} S\xe4tze {'(dry-run)' if args.dry_run else f'→ {OUT_FILE}'}")


if __name__ == "__main__":
    main()
