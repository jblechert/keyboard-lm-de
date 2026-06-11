#!/usr/bin/env python3
"""
Lädt VTT-Transkripte von Podcasts herunter und extrahiert Sätze.

Zwei Strategien:
  - NumberedPodcast: sequentielle Episodennummern, stoppt nach N Fehlern in Folge
  - RssPodcast:      RSS-Feed → VTT-URLs (via podcast:transcript-Tag oder Link + Suffix)

Output: data/raw/{podcast.id}_{podcast.lang}.txt   (Append-Modus)

Usage:
  .venv_ml/bin/python 16_download_podcasts.py [--lang de] [--podcast lnp,raumzeit] [--dry-run]
"""

import argparse
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Gemeinsame VTT-Logik ──────────────────────────────────────────────────────

def fetch(url: str, timeout: int = 20) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code in (301, 404):
            return None
        print(f"  HTTP {e.code}: {url[:70]}")
        return None
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


# ── Podcast-Typen ─────────────────────────────────────────────────────────────

@dataclass
class NumberedPodcast:
    """Episoden per sequentielle Nummer abrufbar (Podlove-Transcript-URL)."""
    id: str
    name: str
    lang: str
    url_fn: Callable[[int], str]   # n → URL
    prefix: str                     # Label für Ausgabe, z.B. "LNP"
    start: int = 1
    max_ep: int = 9999
    consecutive_stop: int = 5
    delay: float = 0.3

    @property
    def out_file(self) -> Path:
        return Path(f"data/raw/{self.id}_{self.lang}.txt")

    def download(self, dry_run: bool) -> int:
        total = 0
        consecutive = 0
        with (self.out_file.open("a", encoding="utf-8") if not dry_run
              else open("/dev/null", "w")) as fout:
            for n in range(self.start, self.max_ep + 1):
                url = self.url_fn(n)
                content = fetch(url)
                if not content or not content.strip().startswith("WEBVTT"):
                    consecutive += 1
                    if consecutive >= self.consecutive_stop:
                        print(f"  {self.consecutive_stop} Fehler in Folge — "
                              f"stoppe bei {self.prefix}{n:03d}")
                        break
                    continue
                consecutive = 0
                sentences = parse_vtt(content)
                print(f"  {self.prefix}{n:03d}: {len(sentences)} Sätze")
                for s in sentences:
                    fout.write(s + "\n")
                total += len(sentences)
                time.sleep(self.delay)
        return total


@dataclass
class RssPodcast:
    """Episoden via RSS-Feed; VTT-URL entweder im Feed-Tag oder als Link + Suffix."""
    id: str
    name: str
    lang: str
    rss_url: str
    # "transcript_tag": VTT-URL aus <podcast:transcript url="...vtt"> im Feed
    # "link_suffix":    VTT-URL = Episodenlink + vtt_suffix
    vtt_source: str
    link_pattern: str = ""         # Regex für Episodenlink (bei link_suffix)
    vtt_suffix: str = "?podlove_transcript=webvtt"
    delay: float = 0.3

    @property
    def out_file(self) -> Path:
        return Path(f"data/raw/{self.id}_{self.lang}.txt")

    def _parse_rss(self, feed: str) -> list[tuple[str, str | None]]:
        items = re.findall(r"<item>(.*?)</item>", feed, re.DOTALL)
        result = []
        for item in items:
            title_m = re.search(r"<title><!\[CDATA\[(.*?)\]\]>|<title>(.*?)</title>", item)
            title = ((title_m.group(1) or title_m.group(2) or "?").strip()
                     if title_m else "?")
            if self.vtt_source == "transcript_tag":
                vtt_m = re.search(r'podcast:transcript url="(https://[^"]+\.vtt)"', item)
                result.append((title, vtt_m.group(1).strip() if vtt_m else None))
            else:  # link_suffix
                link_m = re.search(self.link_pattern, item)
                if not link_m:
                    continue
                link = link_m.group(0).strip().rstrip("/")
                result.append((title, link + "/" + self.vtt_suffix.lstrip("/")))
        return result

    def download(self, dry_run: bool) -> int:
        print("Lade RSS-Feed …")
        feed = fetch(self.rss_url, timeout=30)
        if not feed:
            print("  Feed nicht erreichbar.")
            return 0

        episodes = self._parse_rss(feed)
        print(f"  {len(episodes)} Episoden im Feed\n")

        total = 0
        with (self.out_file.open("a", encoding="utf-8") if not dry_run
              else open("/dev/null", "w")) as fout:
            for i, (title, vtt_url) in enumerate(episodes):
                short = title[:55]
                if not vtt_url:
                    print(f"  [{i+1:3d}/{len(episodes)}] {short} — kein Transkript")
                    continue
                vtt = fetch(vtt_url)
                if not vtt or not vtt.strip().startswith("WEBVTT"):
                    print(f"  [{i+1:3d}/{len(episodes)}] {short} — kein VTT")
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
                time.sleep(self.delay)
        return total


# ── Podcast-Verzeichnis ───────────────────────────────────────────────────────
# Struktur: lang → Liste von Podcast-Objekten
# Mehrere Einträge mit gleicher id+lang schreiben in dieselbe Datei (Append).

PODCASTS: dict[str, list] = {
    "de": [
        NumberedPodcast(
            id="lnp", name="Logbuch:Netzpolitik", lang="de",
            url_fn=lambda n: f"https://logbuch-netzpolitik.de/lnp{n:03d}?podlove_transcript=webvtt",
            prefix="LNP", start=6,
        ),
        NumberedPodcast(
            id="raumzeit", name="Raumzeit", lang="de",
            url_fn=lambda n: f"https://raumzeit-podcast.de/rz{n:03d}?podlove_transcript=webvtt",
            prefix="RZ", start=1,
        ),
        NumberedPodcast(
            id="parlamentsrevue", name="Parlamentsrevue (PR)", lang="de",
            url_fn=lambda n: f"https://parlamentsrevue.de/wp-content/upload/PR{n:03d}.vtt",
            prefix="PR", start=22, delay=0.5,
        ),
        NumberedPodcast(
            id="parlamentsrevue", name="Parlamentsrevue (LTR)", lang="de",
            url_fn=lambda n: f"https://parlamentsrevue.de/wp-content/upload/LTR{n:03d}.vtt",
            prefix="LTR", start=1, delay=0.5,
        ),
        RssPodcast(
            id="minkorrekt", name="Methodisch Inkorrekt", lang="de",
            rss_url="https://minkorrekt.podigee.io/feed/mp3",
            vtt_source="transcript_tag",
        ),
        RssPodcast(
            id="forschergeist", name="Forschergeist", lang="de",
            rss_url="https://forschergeist.de/feed/mp3",
            vtt_source="link_suffix",
            link_pattern=r"https://forschergeist\.de/podcast/[^\s<\"]+",
            vtt_suffix="?podlove_transcript=webvtt",
        ),
        RssPodcast(
            id="cre", name="CRE: Technik, Kultur, Gesellschaft", lang="de",
            rss_url="https://cre.fm/feed/mp3",
            vtt_source="link_suffix",
            link_pattern=r"https://cre\.fm/cre\d+[^\s<\"]*",
            vtt_suffix="?podlove_transcript=webvtt",
        ),
    ],
}


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="de",
                        help="Sprache (default: de)")
    parser.add_argument("--podcast", default="",
                        help="Komma-getrennte Podcast-IDs (default: alle)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nichts schreiben, nur zählen")
    args = parser.parse_args()

    podcasts = PODCASTS.get(args.lang, [])
    if not podcasts:
        print(f"Keine Podcasts für Sprache '{args.lang}' konfiguriert.")
        return

    filter_ids = {s.strip() for s in args.podcast.split(",") if s.strip()}
    if filter_ids:
        podcasts = [p for p in podcasts if p.id in filter_ids]
        if not podcasts:
            print(f"Keine Podcasts für IDs {filter_ids} gefunden.")
            return

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for podcast in podcasts:
        print(f"\n=== {podcast.name} ===")
        n = podcast.download(dry_run=args.dry_run)
        label = "(dry-run)" if args.dry_run else f"→ {podcast.out_file}"
        print(f"Gesamt: {n:,} Sätze {label}")
        grand_total += n

    if len(podcasts) > 1:
        print(f"\n{'='*40}\nGesamt alle Podcasts: {grand_total:,} Sätze")


if __name__ == "__main__":
    main()
