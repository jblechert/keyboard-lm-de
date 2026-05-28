#!/usr/bin/env python3
"""
Bereinigt Trainingsdaten von unerwünschten Wörtern/Phrasen.

Füge neue unerwünschte Muster zur BANNED-Liste hinzu und führe das Skript
erneut aus — es filtert alle gefundenen Quelldateien in-place.

Unterstützt auch --high-ppl FILE: Liest eine Datei mit Sätzen (eine pro Zeile)
und entfernt exakt diese Sätze aus den Quelldateien.

Usage:
  .venv_ml/bin/python 10_clean_training_data.py [--dry-run]
  .venv_ml/bin/python 10_clean_training_data.py --high-ppl data/high_ppl.txt [--dry-run]
"""

import argparse
import hashlib
import html
import os
import re
import sys
from pathlib import Path

# ── Unerwünschte Muster ───────────────────────────────────────────────────────
# Jeder Eintrag: (pattern, beschreibung)
# Groß-/Kleinschreibung wird ignoriert (re.IGNORECASE).
# Ganzes Wort via \b oder freier Substring — je nach Eintrag.

BANNED = [
    # Überrepräsentierte Einzelwörter
    (r"\bmahd\b",       "Mahd (fälschlicherweise häufig vorhergesagt)"),

    # Nicht-lateinische Schriftsysteme (eindeutig nicht Deutsch)
    (r"[Ѐ-ӿ]",    "Kyrillisch"),
    (r"[؀-ۿ]",    "Arabisch"),
    (r"[一-鿿]",    "CJK-Zeichen (Chinesisch/Japanisch)"),
    (r"[가-힯]",    "Hangul (Koreanisch)"),
    (r"[ऀ-ॿ]",    "Devanagari"),

    # Alte deutsche Rechtschreibung / Scanfehler (typisch für schlecht gefilterte Webdaten)
    (r"\bae(?:hn|uss|uss)\w*", "Alte Schreibweise ae- (Aehnliche, Aeusserst …)"),
    (r"\bue(?:ber|brig|bel)\w*", "Alte Schreibweise ue- (Ueber, Uebrig …)"),

    # Spam-Muster: viele Pipe- oder Tabulator-getrennten Felder → Datentabellen
    (r"(?:\||\t){3,}",      "Tabellenzeile (≥3 Pipes/Tabs) — kein Fließtext"),

    # Strukturierte Produktdaten / Key-Value-Spam (Hotel, Shop, Immobilien …)
    (r"(?:[^:]*:){3}",      "≥3 Doppelpunkte (Produkt-/Listing-Spam)"),

    # Sehr viele aufeinanderfolgende Sonderzeichen → Listenspam
    (r"[*•·]{4,}",          "Bullet-Spam (≥4 Sonderzeichen in Folge)"),

    # Schlechte Satzenden
    (r"\.{2,}\s*$",         "Satzende mit .. oder ... (unvollständig/abgebrochen)"),
    (r"[!?]{2,}\s*$",       "Satzende mit !! oder ?? (Ausrufe-Spam)"),

    # Zeilen die mit [Tag] beginnen → Blog-Kategorien, Subreddit-Flair, Bildunterschriften
    (r"^\[",                 "Zeile beginnt mit [ (Blog-Tag, Kategorie-Header)"),

    # Alphanumerische Codes → Produkt-SKUs, Hashes, Referenznummern
    (r"\b[a-zA-Z0-9]*\d{3,}[a-zA-Z][a-zA-Z0-9-]*\b",
     "Alphanumerischer Code (SKU, Hash, Referenznummer)"),

    # PDF-Hinweise und Autoren-Katalogeinträge → Buchdatenbank-Spam
    (r"\bPDF\b",             "PDF-Verweis (Buchdatenbank, Downloadseite)"),
    (r"\bBy author\b",       "Englischer Autorenhinweis (Katalog-Eintrag)"),

    # Text-Emojis / Emoticons → informeller Forenstil
    (r"(?:[:;=]-?[)D(\|PpOo\/\\]|\^\^+|[xX][Dd](?!\w)|<3)",
     "Text-Emoji / Emoticon (:D, ^^, xD, <3 …)"),

    # Konkrete Datumsangaben → Event-Ankündigungen, Kalendereinträge
    (r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
     "Numerisches Datum (dd.mm.yyyy)"),
    (r"\b\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}\b",
     "Ausgeschriebenes Datum (dd. Monat yyyy)"),
    # Falsche Grossbuchstaben nach Umlaut (SEO-Spam): "SorgfäLtig", "FußBall"
    (r"[äöüß][A-ZÄÖÜ]",
     "Umlaut gefolgt von Grossbuchstabe (SEO-Spam)"),

    # CamelCase-Zusammenschreibungen: "SätzeDieNurAusSoEtwasBestanden"
    (r"(?:[a-zäöüß][A-ZÄÖÜ]){3,}",
     "CamelCase-Run-on (3+ Uebergaenge, zusammengeschriebene Woerter)"),

    # Hashtag-Ansammlungen -> Social-Media-Metadaten
    (r"(?:.*#){3}",          ">=3 Hashtags (Social-Media-Tag-Spam)"),

    # Breadcrumb-Navigation -> Website-Navigationspfade
    (r"(?:[^>]*>){3}",       ">=3 > (Breadcrumb-Navigation)"),

    # Bildunterschriften und Fotokredite -> redaktionelle Metadaten
    (r"^\s*Bild\w*\s*\w*:",    "Bildunterschrift (Bild oben:, Bild links: ...)"),
    (r"FOTOS?:",                  "Fotokredit (FOTO:, FOTOS: ...)"),

    # Bracket-Navigation-Tags: [Top], [Kategorie], [ Bearbeiten ] etc.
    (r"\[[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß]{1,20}\]",
     "Bracket-Tag ([Top], [Kategorie] ...) -- Navigation/Template-Artefakt"),
    (r"\[\s+[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß]+\s+\]",
     "Bracket-Tag mit Leerzeichen ([ Bearbeiten ]) -- Wikipedia-Edit-Link"),

    # Adresszeilen: Komma vor PLZ oder Strassenname + Hausnummer
    (r",\s*\d{5}\b",
     "Adresse (Komma vor Postleitzahl)"),
    (r"\b(?:Stra\xdfe|Gasse|Weg|Allee|Platz|Ring|Damm|Chaussee)\s+\d+\b",
     "Adresse (Strassenname + Hausnummer)"),

    # E-Commerce-Spam: Shop-Seiten, Produktlistings
    (r"\bVersandkosten\b",  "Versandkosten (Shop-Listing)"),
    (r"\bMwSt\.?\b",        "MwSt (Shop-Listing)"),
    (r"\bWarenkorb\b",      "Warenkorb (Shop-Listing)"),
    (r"\bzzgl\.\b",         "zzgl. (Preis-Listing)"),

    # Haekchen-Symbole: Feature-Listen, Buchungsbestaetigungen, Produktbeschreibungen
    (r"[\u2713\u2714\u2611\u2705]", "Haekchen-Symbol (\u2713\u2714\u2611\u2705)"),
]

# ── Ersetzungen ───────────────────────────────────────────────────────────────
# Jeder Eintrag: (compiled_pattern, replacement, beschreibung)
# Werden auf jede behaltene Zeile angewendet (in Reihenfolge).

def _build_replacements():
    pairs = [
        # Schweizer Schreibweise ohne ß → Standard-Deutsch
        (r"\bgrosse(n|r|s|m)?\b", lambda m: "große" + (m.group(1) or ""),          "grosse→große"),
        (r"\bGrosse(n|r|s|m)?\b", lambda m: "Große" + (m.group(1) or ""),          "Grosse→Große"),
        # Mojibake: doppelt falsch dekodierte Umlaute reparieren
        ("Ã¤", "ä", "Mojibake Ã¤ -> ä"),
        ("Ã¶", "ö", "Mojibake Ã¶ -> ö"),
        ("Ã¼", "ü", "Mojibake Ã¼ -> ü"),
        ("Ã", "Ä", "Mojibake Ã -> Ä"),
        ("Ã", "Ö", "Mojibake Ã -> Ö"),
        ("Ã", "Ü", "Mojibake Ã -> Ü"),
        ("Ã", "ß", "Mojibake Ã -> ß"),
        # C1-Steuerzeichen (U+0080-U+009F): Windows-1252-Artefakte
        ("[\x80-\x9f]", "", "C1-Steuerzeichen entfernen (Windows-1252-Artefakte)"),
        # Mojibake Typ 2: UTF-8-Bytes als Windows-1252 gelesen (ae-Euro-Sequenzen)
        ("â€ž", "",  "Mojibake ae-Euro-z -> Anf.-Zeichen (wird gestrichen)"),
        ("â€œ", "",  "Mojibake ae-Euro-oe -> Anf.-Zeichen (wird gestrichen)"),
        ("â€“", "-",  "Mojibake ae-Euro-lq -> Gedankenstrich"),
        ("â€’", "'", "Mojibake ae-Euro-rq -> Apostroph"),
        ("â€™", "",  "Mojibake ae-Euro-tm -> Anf.-Zeichen (wird gestrichen)"),
        # Anführungszeichen aller Art entfernen
        (r'["“”„«»]', "",  'Anführungszeichen entfernen ("„"»«)'),
        # Zitatmarker und Pfeilzeichen entfernen: > Zitat, <Verweis>
        (r'[<>]', '', 'Spitze Klammern (Zitatmarker, Pfeil)'),
        # Fehlende Leerzeichen nach Satzzeichen reparieren
        # Punkt: Prof.Dr. -> Prof. Dr., e.V. -> e. V., Satz.Satz -> Satz. Satz
        (r'([a-zäöüß])\.([A-ZÄÖÜ])', r'\1. \2',
         'Fehlender Abstand nach Punkt (Prof.Dr. -> Prof. Dr.)'),
        # Ausrufe-/Fragezeichen: Toll!Super -> Toll! Super
        (r'([a-zäöüß])([!?])([A-ZÄÖÜ])', r'\1\2 \3',
         'Fehlender Abstand nach ! oder ? (Toll!Super -> Toll! Super)'),
        # Semikolon: wort;Wort -> wort; Wort
        (r'([a-zäöüß]);([A-Za-zÄÖÜäöü])', r'\1; \2',
         'Fehlender Abstand nach Semikolon'),
    ]
    return [(re.compile(p), repl, desc) for p, repl, desc in pairs]

REPLACEMENTS = _build_replacements()

# ── Quelldateien ──────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/tatoeba_de.txt"),
    Path("data/c4_de.txt"),
    *sorted(Path("data").glob("synthetic_*.txt")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur zählen, nichts schreiben")
    parser.add_argument("--dedup", action="store_true",
                        help="Duplikate entfernen (hash-basiert, ueber alle Quelldateien)")
    parser.add_argument("files", nargs="*", metavar="FILE",
                        help="Zu verarbeitende Dateien (Standard: alle SOURCES)")
    parser.add_argument("--high-ppl", metavar="FILE",
                        help="Datei mit Sätzen hoher Perplexity (eine pro Zeile); "
                             "exakte Übereinstimmungen werden entfernt")
    args = parser.parse_args()

    patterns = [(re.compile(p, re.IGNORECASE), desc) for p, desc in BANNED]

    # Optionale Menge von Sätzen aus --high-ppl
    high_ppl_set: set[str] = set()
    if args.high_ppl:
        ppl_path = Path(args.high_ppl)
        if not ppl_path.exists():
            print(f"Fehler: --high-ppl Datei nicht gefunden: {ppl_path}", file=sys.stderr)
            sys.exit(1)
        high_ppl_set = {l.strip() for l in ppl_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"{len(high_ppl_set)} Sätze aus {ppl_path} geladen (hohe Perplexity).")

    sources = [Path(f) for f in args.files] if args.files else SOURCES
    seen_hashes: set[int] = set()
    total_removed = 0

    for src in sources:
        if not src.exists():
            continue

        tmp = src.with_suffix(".tmp")
        removed_counts = {desc: 0 for _, desc in BANNED}
        n_ppl_removed = 0
        n_kept = 0
        n_total = 0
        n_dedup_removed = 0
        replacement_counts = {desc: 0 for _, _, desc in REPLACEMENTS}

        with src.open("r", encoding="utf-8") as fin, \
             (tmp.open("w", encoding="utf-8") if not args.dry_run else open(os.devnull, "w")) as fout:
            for line in fin:
                n_total += 1
                stripped = line.strip()

                if stripped in high_ppl_set:
                    n_ppl_removed += 1
                    continue

                if args.dedup:
                    h = int.from_bytes(hashlib.md5(stripped.encode()).digest()[:8], "little")
                    if h in seen_hashes:
                        n_dedup_removed += 1
                        continue
                    seen_hashes.add(h)

                drop = False
                for pat, desc in patterns:
                    if pat.search(line):
                        removed_counts[desc] += 1
                        drop = True
                        break

                if not drop:
                    for pat, repl, desc in REPLACEMENTS:
                        new_line, n = pat.subn(repl, line)
                        if n:
                            replacement_counts[desc] += n
                            line = new_line
                    fout.write(line)
                    n_kept += 1

                if n_total % 5_000_000 == 0:
                    print(f"  {src.name}: {n_total:,} gelesen …", flush=True)

        n_removed = n_total - n_kept
        total_removed += n_removed

        if n_removed == 0:
            print(f"{src}: keine Treffer ({n_total:,} Sätze)")
            if not args.dry_run:
                tmp.unlink()
            continue

        print(f"{src}: {n_removed:,} Sätze entfernt (von {n_total:,})")
        if n_ppl_removed:
            print(f"  {n_ppl_removed:>8,}×  hohe Perplexity (--high-ppl)")
        if n_dedup_removed:
            print(f"  {n_dedup_removed:>8,}×  Duplikat (--dedup)")
        for desc, count in removed_counts.items():
            if count:
                print(f"  {count:>8,}×  {desc}")

        for desc, count in replacement_counts.items():
            if count:
                print(f"  {count:>8,}×  {desc}")

        if not args.dry_run:
            tmp.replace(src)

    print(f"\nGesamt entfernt: {total_removed:,}")
    if args.dry_run:
        print("(Dry-run — keine Datei verändert)")


if __name__ == "__main__":
    main()
