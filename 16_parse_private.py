#!/usr/bin/env python3
"""
Extrahiert Nachrichtentext aus Signal- und WhatsApp-Desktop-Exporten.

Signal:    Copy-Paste aus Signal Desktop (ohne strukturiertes Format)
WhatsApp:  Copy-Paste aus WhatsApp Web/Desktop: [14.5., 00:09] Name: Text

Das Format wird automatisch erkannt. Ausgabe: eine Nachricht pro Zeile.

Usage:
  .venv_ml/bin/python 16_parse_private.py data/private-signal.txt
  .venv_ml/bin/python 16_parse_private.py data/private-whatsapp.txt
  .venv_ml/bin/python 16_parse_private.py data/private-*.txt --output data/private_de.txt
  .venv_ml/bin/python 16_parse_private.py data/private-signal.txt --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

# ── WhatsApp ──────────────────────────────────────────────────────────────────
# Format: [14.5., 00:09] Name: Nachrichtentext
WA_LINE_RE = re.compile(r"^\[\d{1,2}\.\d{1,2}\.(?:\d{4})?,\s*\d{2}:\d{2}\]\s*[^:]+:\s*(.+)$")
WA_DETECT_RE = re.compile(r"^\[\d{1,2}\.\d{1,2}")

# ── Signal ────────────────────────────────────────────────────────────────────
SIGNAL_SKIP = [
    re.compile(r"^\d{1,2}:\d{2}$"),
    re.compile(r"^(Mo|Di|Mi|Do|Fr|Sa|So)\.$"),
    re.compile(r"^\w{2}\.,\s+\d{1,2}\.\s+\w+"),
    re.compile(r"^Du$"),
    re.compile(r"^(Video|Bild|Audio|GIF|Datei|Foto|Sprachnachricht)$", re.I),
    re.compile(r"Tabs verbergen|Chats|Anrufe|Storys|Einstellungen"),
    re.compile(r"Spende|Signal braucht|Jetzt nicht|Nach Ungelesen"),
    re.compile(r"Keine gemeinsamen Gruppen|Neuer Chat|More Actions"),
    re.compile(r"^\s*$"),
]
CONTACT_NAME_RE = re.compile(r"[⁨⁩]")

# ── Gemeinsam ─────────────────────────────────────────────────────────────────
MIN_LEN = 8
MAX_LEN = 500

QUALITY_SKIP = [
    re.compile(r"https?://\S+"),                       # URLs (auch mitten im Satz)
    re.compile(r"\bwww\.\S+\.\w{2,}"),                # www. URLs
    re.compile(r"^[%\\\/\w]+\\"),                      # Windows-Pfade
    re.compile(r"^\W+$"),                              # nur Sonderzeichen / Emojis
]


def is_german_text(line: str) -> bool:
    if not re.search(r"[a-zA-ZäöüÄÖÜß]", line):
        return False
    for pat in QUALITY_SKIP:
        if pat.match(line.strip()):
            return False
    return True


# ── WhatsApp Parser ───────────────────────────────────────────────────────────

def parse_whatsapp(path: Path) -> list[str]:
    messages = []
    current: list[str] = []

    def flush():
        if current:
            msg = " ".join(current).strip()
            if MIN_LEN <= len(msg) <= MAX_LEN and is_german_text(msg):
                messages.append(msg)
            current.clear()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = WA_LINE_RE.match(line)
            if m:
                flush()
                text = m.group(1).strip()
                # Medien-Platzhalter überspringen
                if re.match(r"^(Bild|Video|Audio|GIF|Datei|Dokument|Sticker|weggelassen)(\s*$|,)", text, re.I):
                    continue
                current.append(text)
            elif current and line.strip():
                # Fortsetzung einer mehrzeiligen Nachricht
                current.append(line.strip())
    flush()
    return messages


# ── Signal Parser ─────────────────────────────────────────────────────────────

def parse_signal(path: Path) -> list[str]:
    messages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if CONTACT_NAME_RE.search(stripped):
                continue
            if any(pat.search(stripped) for pat in SIGNAL_SKIP):
                continue
            if not (MIN_LEN <= len(stripped) <= MAX_LEN):
                continue
            if not is_german_text(stripped):
                continue
            messages.append(stripped)
    return messages


# ── Nextcloud Talk / Plain Parser ─────────────────────────────────────────────

def parse_plain(path: Path) -> list[str]:
    messages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped == "{file}":
                continue
            if not (MIN_LEN <= len(stripped) <= MAX_LEN):
                continue
            if not is_german_text(stripped):
                continue
            messages.append(stripped)
    return messages


# ── Auto-Detect ───────────────────────────────────────────────────────────────

def detect_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 40:
                break
            stripped = line.strip()
            if WA_DETECT_RE.match(stripped):
                return "whatsapp"
            if CONTACT_NAME_RE.search(stripped):
                return "signal"
    return "plain"


def extract(path: Path) -> list[str]:
    fmt = detect_format(path)
    print(f"{path.name}: Format erkannt als '{fmt}'", file=sys.stderr)
    if fmt == "whatsapp":
        return parse_whatsapp(path)
    if fmt == "signal":
        return parse_signal(path)
    return parse_plain(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("data/private_de.txt"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_messages: list[str] = []
    for path in args.files:
        if not path.exists():
            print(f"Warnung: {path} nicht gefunden", file=sys.stderr)
            continue
        msgs = extract(path)
        print(f"  → {len(msgs)} Nachrichten", file=sys.stderr)
        all_messages.extend(msgs)

    seen: set[str] = set()
    deduped = [m for m in all_messages if not (m in seen or seen.add(m))]
    print(f"Gesamt: {len(deduped)} ({len(all_messages)-len(deduped)} Duplikate entfernt)",
          file=sys.stderr)

    if args.dry_run:
        print("\n--- Vorschau (erste 20) ---")
        for m in deduped[:20]:
            print(m)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for m in deduped:
            f.write(m + "\n")
    print(f"→ {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
