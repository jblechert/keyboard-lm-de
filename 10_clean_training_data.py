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
# Jeder Eintrag: (pattern, beschreibung[, flags])
# Standard-Flags: re.IGNORECASE. Drittes Element = explizite Flags (z.B. 0).

# Sprach-unabhängige Web-Artefakte: Spam, Struktur, Format-Müll
WEB_ARTIFACTS = [
    # Tabellenzeilen, Listings
    (r"(?:\||\t){3,}",      "Tabellenzeile (≥3 Pipes/Tabs) — kein Flie\xdftext"),
    (r"(?:[^:]*:){3}",      "≥3 Doppelpunkte (Produkt-/Listing-Spam)"),
    (r"[*•\xb7]{4,}",  "Bullet-Spam (≥4 Sonderzeichen in Folge)"),

    # Schlechte Satzenden
    (r"(?:\.{2,}|\u2026)\s*$", "Satzende mit .. / ... / … (unvollständig/abgebrochen)"),
    (r"\[\.\.\.|\[…\]", "Eckige Klammer mit Auslassungspunkten ([...]/[…])"),
    (r"[!?]{2,}\s*$",       "Satzende mit !! oder ?? (Ausrufe-Spam)"),
    (r"\s(?:Dr|Prof|Hr|Fr|Nr|Col|Gen|Lt|Cpt|St|Tel|Bd|Mio|Mrd|Jh|bzw|ggf|inkl|usw|vgl|vs|Abb|Abs|ca|max|min)\.\s*$",
     "Satzende mit Abkürzung (abgeschnittener Satz)"),
    (r"\s(?:[1-9]|[12]\d|3[01])\.\s*$",
     "Satzende mit Tagesdatum (abgeschnittener Satz: vom 22., bis zum 7.)"),

    # Strukturelle Artefakte
    (r"^-\s+\S",            "Zeile beginnt mit Listenpunkt (- item)"),
    (r"^[•▪▸►]",              "Zeile beginnt mit Aufzählungszeichen (•▪▸►)"),
    (r"^\[",                 "Zeile beginnt mit [ (Blog-Tag, Kategorie-Header)"),
    (r"(?:.*#){3}",          "≥3 Hashtags (Social-Media-Tag-Spam)"),
    (r"\?[a-zäöüß]{2}",     "Fragezeichen statt ß (Kodierungsfehler: Bekannterma?en)"),
    (r"^(?:(?![äöüÄÖÜß]).)*\b(?:the|find|visit|recent|your|with|from|that|this|have|will|are|were|been|their|they|what|which|when|where|how|you|our|more|also|most|some|all|can|not|for|its|but|and|has|was|his|her)\b(?:(?![äöüÄÖÜß]).)*$",
     "Englischer Satz (keine Umlaute + englische Funktionswörter)", re.IGNORECASE),
    (r"(?:[^>]*>){3}",       "≥3 > (Breadcrumb-Navigation)"),
    (r"\u2192|\u279c|\u27a1", "HTML-Navigationspfeil (→ ➜ ➡ in Linktext)"),
    (r"\[[A-Z\xc4\xd6\xdc][a-zA-Z\xc4\xd6\xdc\xe4\xf6\xfc\xdf]{1,20}\]",
     "Bracket-Tag ([Top], [Kategorie]) — Navigation/Template-Artefakt"),
    (r"\[\s+[A-Z\xc4\xd6\xdc][a-zA-Z\xc4\xd6\xdc\xe4\xf6\xfc\xdf]+\s+\]",
     "Bracket-Tag mit Leerzeichen ([ Bearbeiten ]) — Wikipedia-Edit-Link"),

    # Codes und Referenznummern
    (r"\b[a-zA-Z0-9]*\d{3,}[a-zA-Z][a-zA-Z0-9-]*\b",
     "Alphanumerischer Code (SKU, Hash, Referenznummer)"),
    (r"\bPDF\b",             "PDF-Verweis (Buchdatenbank, Downloadseite)"),
    (r"\bBy author\b",       "Englischer Autorenhinweis (Katalog-Eintrag)"),

    # Emojis und Symbole
    (r"(?:[:;=]-?[)D(\|PpOo\/\\]|\^\^+|[xX][Dd](?!\w)|<3)",
     "Text-Emoji / Emoticon (:D, ^^, xD, <3 …)"),
    ("[\U0001F300-\U0001FAFF☀-➿⌀-⏿]",
     "Unicode-Emoji (\U0001F642\U0001F389♥ …)"),
    (r"[✓✔☑✅]", "H\xe4kchen-Symbol (✓✔☑✅)"),

    # Fotokredit (strukturell, nicht sprachspezifisch)
    (r"FOTOS?:",              "Fotokredit (FOTO:, FOTOS: ...)"),

    # CamelCase-Zusammenschreibungen: "SätzeDieNurAusSoEtwasBestanden"
    (r"\S*[a-z\xe4\xf6\xfc\xdf][A-Z\xc4\xd6\xdc]\S*[a-z\xe4\xf6\xfc\xdf][A-Z\xc4\xd6\xdc]\S*",
     "CamelCase-Run-on (2+ \xdcberg\xe4nge in einem Token)", 0),

    # Datum: numerisch (dd.mm.yyyy) — sprachübergreifend
    (r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
     "Numerisches Datum (dd.mm.yyyy)"),

    # Keyword-Spam: Wortwiederholung
    # Fängt "Zierschotter Zierschotter" (direkt) und "esszimmer - esszimmer" (mit Trennzeichen)
    (r"\b(\w{4,})\s+\1\b",
     "Direktes Doppelwort (Keyword-Spam: Zierschotter Zierschotter)"),
    (r"\b(\w{7,})\b\W{1,5}\b\1\b",
     "Nahes Doppelwort mit Trennzeichen (Keyword-Spam: esszimmer - esszimmer)"),
]

# Deutsch-spezifische Filter: falsche Schreibweisen, DE-E-Commerce, DE-Adressen
LANG_DE = [
    # Überrepräsentierte Einzelwörter
    (r"\bmahd\b",       "Mahd (f\xe4lschlicherweise h\xe4ufig vorhergesagt)"),

    # Alte deutsche Rechtschreibung / Scanfehler
    (r"\bae(?:hn|uss|uss)\w*", "Alte Schreibweise ae- (\xc4hnliche, \xc4u\xdferst …)"),
    (r"\bue(?:ber|brig|bel)\w*", "Alte Schreibweise ue- (\xdcber, \xdcbrig …)"),

    # SEO-Spam: Umlaut gefolgt von Großbuchstabe — nie korrekte deutsche Grammatik
    (r"[\xe4\xf6\xfc\xdf][A-Z\xc4\xd6\xdc]",
     "Umlaut gefolgt von Gro\xdfbuchstabe (SEO-Spam)", 0),

    # Ausgeschriebene deutsche Datumsangaben
    (r"\b\d{1,2}\.\s*(?:Januar|Februar|M\xe4rz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}\b",
     "Ausgeschriebenes Datum (dd. Monat yyyy)"),

    # Adressen
    (r",\s*\d{5}\b",
     "Adresse (Komma vor Postleitzahl)"),
    (r"\b(?:Stra\xdfe|Gasse|Weg|Allee|Platz|Ring|Damm|Chaussee)\s+\d+\b",
     "Adresse (Stra\xdfenname + Hausnummer)"),

    # E-Commerce-Spam
    (r"\bVersandkosten\b",  "Versandkosten (Shop-Listing)"),
    (r"\bMwSt\.?\b",        "MwSt (Shop-Listing)"),
    (r"\bWarenkorb\b",      "Warenkorb (Shop-Listing)"),
    (r"\bzzgl\.\b",         "zzgl. (Preis-Listing)"),

    # Amazon / Buchhandel-Spam
    (r"\bGebundene Ausgabe\b",  "Amazon-Format (Gebundene Ausgabe)"),
    (r"\bBestseller-Rang\b",    "Amazon-Rang (Bestseller-Rang)"),
    (r"\bFremdsprachige B\xfccher\b", "Amazon-Kategorie (Fremdsprachige B\xfccher)"),
    (r"\bSiehe Top 100\b",      "Amazon-Rang-Link (Siehe Top 100)"),
    (r"\bEUR\s+\d",             "Preis-Listing (EUR 12,34)"),
    (r"Zur\xfcck\s*Weiter",     "Amazon-Navigation (Zur\xfcckWeiter)"),
    (r"\bAngebote\s+ab\b",      "Amazon-Preis (Angebote ab EUR)"),

    # Medien-Produktlistings
    (r"\((?:CD|DVD|Blu-?[Rr]ay|Buch|H\xf6rbuch|H\xf6rbuch MP3|MP3|EP|LP|VHS|Vinyl|Book|Film|Movie|Album|Novel|Series|Comic)\)\s*$",
     "Medien-Produktlisting (Buch/CD/DVD/Book/Film am Zeilenende)"),
    (r"\b(?:Doppel|Triple|Single|Einzel)-?(?:CD|DVD|LP)\b",
     "Medien-Produktformat (Doppel-CD etc.)"),

    # Hotel-/Reise-Spam
    (r"\bNo:\s*\d",  "Englische Hausnummer (No:9 — Hotel-/Adress-Listing)"),
    (r"\b(?:un|ver|be|ge|ent|emp|zer)\s*$",
     "Abgeschnittenes Pr\xe4fix am Zeilenende (un/ver/ge/be/ent — Truncation)", 0),

    # Blog-Metadaten
    (r"keine Kommentare",    "Blog-Metadaten (keine Kommentare)"),
    (r"geschrieben von\b",   "Blog-Autorenzeile (geschrieben von ...)"),
    (r"\bklicken?\s+(?:Sie\s+)?(?:hier|auf|bitte)\b|\bhier\s+klicken\b",
     "Navigations-CTA (klicken Sie hier, hier klicken, klick auf …)"),
    (r"^\s*Bild\w*\s*\w*:",  "Bildunterschrift (Bild oben:, Bild links: ...)"),
    (r"^\s*Tags?\s*(?:für\s+diesen\s+Artikel\s*)?:", "Blog-Tags (Tags: keyword, keyword …)"),
    (r"^\s*(?:Regie|Drehbuch|Darsteller|Produktion|Kamera|Schnitt|Musik)\s*:",
     "Film-/TV-Credits (Regie:, Darsteller:, Drehbuch: …)"),
    (r"^\s*Flüge\s+von\s+\w+\s+nach\s+\w+",
     "Reise-Suchergebnis (Flüge von X nach Y)"),

    # Produktkategorien: synthetisch abgedeckt, in c4 überwiegend E-Commerce-Spam
    (r"\b(?:Nagel|Nägel|Schraube|Schrauben|Dübel|Akkuschrauber|Bohrmaschine|Sägeblatt|Winkelschleifer|Unterlegscheibe)\b",
     "Heimwerker-Produkt (Nägel/Schrauben/Dübel — synthetisch abgedeckt)"),
    (r"\b(?:Badewanne|Duschkabine|Waschtisch|Sanitär|Duschwanne|Badmöbel|Badarmatur)\b",
     "Bad-Produkt (Badewanne/Duschkabine — synthetisch abgedeckt)"),
    (r"\b(?:Sideboard|Schrankwand|Wohnwand|Polsterecke|Kommode|Nachttisch|Kleiderschrank|Wohnzimmerschrank)\b",
     "Möbel-Produkt (Sideboard/Kommode — synthetisch abgedeckt)"),
    (r"\b(?:Felge|Felgen|Stoßstange|Motoröl|Bremsscheibe|Spoiler|Auspuff|Tuning|Kotflügel)\b",
     "Auto-Zubehör (Felgen/Bremsscheibe — synthetisch abgedeckt)"),
    (r"\b(?:Damenjacke|Herrenhose|Damenbluse|Herrenhemd|Übergröße|Konfektionsgröße|Damenmantel|Herrenjacke)\b",
     "Kleidungs-Listing (Damenjacke/Herrenhose — synthetisch abgedeckt)"),

    # Glücksspiel-SEO-Spam: "Beste Spielothek in [Stadt] finden"
    (r"\bSpielothek\b", "Glücksspiel-SEO-Spam (Spielothek in ... finden)"),
    (r"\b(?:bet365|betway|bwin|tipico|betsson|unibet|pokerstars|888casino)\b",
     "Glücksspiel-Anbieter (bet365/bwin/tipico …)"),

    # Kommentarbox-UI
    (r"\d+\s*Zeichen\s*(?:übrig|verbleibend)", "Zeichenzähler (Web-Kommentarbox-UI)"),
    (r"\bSchreiben Sie.*Kommentar\b", "Kommentarbox-Aufforderung"),

    # E-Commerce-Header
    (r"\bBestellen oder Ausleihen\b", "E-Commerce-Header (Kaufen, Bestellen oder Ausleihen)"),
]

# Nicht-deutsche Schriften und Zeichen (Sprach-Ausschlüsse)
LANG_EXCLUDE = [
    # Nicht-deutsche europäische Sonderzeichen: Estnisch/Finnisch/Polnisch etc.
    (r"[õőűčšžāēīūţşąęśźżłń]",
     "Nicht-dt. EU-Buchstabe (\xf5čšžāł — Estnisch/Polnisch/Lettisch …)"),
    # Thorn: nur Isländisch/Altenglisch, nie Deutsch (auch: falsch konvertiertes ß)
    (r"[þÞ]",
     "Thorn (Isländisch/Altenglisch — kein deutsches Zeichen)", 0),
    # Verdoppelte Umlaute: typisch Finnisch/Estnisch, nie Deutsch
    # Shoutout to the fine people of Kouvola — your ää is beautiful, your New Year’s dynamite louder.
    (r"[\xe4\xf6\xfc]{2}",
     "Verdoppelter Umlaut (\xe4\xe4/\xf6\xf6/\xfc\xfc — Finnisch/Estnisch)", 0),
    # Transliteriertes Russisch in lateinischer Schrift
    (r"\b\w*(?:owaja|tscheski|owskij|owuju)\b", "Transliteriertes Russisch (owaja/tscheski/owskij)"),
    (r"[Ѐ-ӿ]",    "Kyrillisch"),
    (r"[؀-ۿ]",    "Arabisch"),
    (r"[一-鿿]",    "CJK-Zeichen (Chinesisch/Japanisch)"),
    (r"[가-힯]",    "Hangul (Koreanisch)"),
    (r"[ऀ-ॿ]",    "Devanagari"),
]

# ── Gesprochene Sprache / Transkripte ─────────────────────────────────────────
# Whisper-spezifische Artefakte aus Podcast-Transkripten
SPEECH_ARTIFACTS = [
    (r"\s+(und|oder|aber|denn|sondern)\.?\s*$",     "Satz endet mit Konjunktion"),
    (r"(\w)\1{3,}",                                   "Stottern/Wiederholung (aaaa)"),
    (r"Ă[^\x00-\x7f]",                               "Unreparabler Mojibake (Ă + Sonderzeichen)"),
]

BANNED = WEB_ARTIFACTS + LANG_DE + LANG_EXCLUDE + SPEECH_ARTIFACTS

# ── Ersetzungen ───────────────────────────────────────────────────────────────
# Jeder Eintrag: (compiled_pattern, replacement, beschreibung)
# Werden auf jede behaltene Zeile angewendet (in Reihenfolge).

def _try_encode(s: str, enc: str) -> bool:
    """Gibt True zurück wenn s.encode(enc).decode('utf-8') einen gültigen deutschen Text ergibt."""
    try:
        s.encode(enc).decode('utf-8')
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def _build_replacements():
    pairs = [
        # Schweizer Schreibweise ohne ß → Standard-Deutsch
        (r"\bgrosse(n|r|s|m)?\b", lambda m: "gro\xdfe" + (m.group(1) or ""),          "grosse→gro\xdfe"),
        (r"\bGrosse(n|r|s|m)?\b", lambda m: "Gro\xdfe" + (m.group(1) or ""),          "Grosse→Gro\xdfe"),
        # Mojibake: doppelt falsch dekodierte Umlaute reparieren
        ("\xc3\xa4", "\xe4", "Mojibake \xc3\xa4 -> \xe4"),
        ("\xc3\xb6", "\xf6", "Mojibake \xc3\xb6 -> \xf6"),
        ("\xc3\xbc", "\xfc", "Mojibake \xc3\xbc -> \xfc"),
        ("\xc3", "\xc4", "Mojibake \xc3 -> \xc4"),
        ("\xc3", "\xd6", "Mojibake \xc3 -> \xd6"),
        ("\xc3", "\xdc", "Mojibake \xc3 -> \xdc"),
        ("\xc3", "\xdf", "Mojibake \xc3 -> \xdf"),
        # C1-Steuerzeichen (U+0080-U+009F): Windows-1252-Artefakte
        ("[\x80-\x9f]", "", "C1-Steuerzeichen entfernen (Windows-1252-Artefakte)"),
        # Mojibake Typ 2: UTF-8-Bytes als Windows-1252 gelesen (ae-Euro-Sequenzen)
        ("\xe2\x80\x9e", "",  "Mojibake ae-Euro-z -> Anf.-Zeichen (wird gestrichen)"),
        ("\xe2\x80\x9c", "",  "Mojibake ae-Euro-oe -> Anf.-Zeichen (wird gestrichen)"),
        ("\xe2\x80\x93", "-",  "Mojibake ae-Euro-lq -> Gedankenstrich"),
        ("\xe2\x80\x91", "'", "Mojibake ae-Euro-rq -> Apostroph"),
        ("\xe2\x80\x99", "",  "Mojibake ae-Euro-tm -> Anf.-Zeichen (wird gestrichen)"),
        # Anführungszeichen aller Art entfernen
        (r'["""„\xab\xbb]', "",  'Anf\xfchrungszeichen entfernen ("„"\xbb\xab)'),
        # Zitatmarker und Pfeilzeichen entfernen: > Zitat, <Verweis>
        (r'[<>]', '', 'Spitze Klammern (Zitatmarker, Pfeil)'),
        # Fehlende Leerzeichen nach Satzzeichen reparieren
        # Weiches Trennzeichen (U+00AD): Formatierungsartefakt aus Web-Quellen
        ("\xad", "", "Weiches Trennzeichen (U+00AD) entfernen"),
        # Punkt: Prof.Dr. -> Prof. Dr., e.V. -> e. V., Satz.Satz -> Satz. Satz
        (r'([a-z\xe4\xf6\xfc\xdf])\.([A-Z\xc4\xd6\xdc])', r'\1. \2',
         'Fehlender Abstand nach Punkt (Prof.Dr. -> Prof. Dr.)'),
        # Ausrufe-/Fragezeichen: Toll!Super -> Toll! Super
        (r'([a-z\xe4\xf6\xfc\xdf])([!?])([A-Z\xc4\xd6\xdc])', r'\1\2 \3',
         'Fehlender Abstand nach ! oder ? (Toll!Super -> Toll! Super)'),
        # Semikolon: wort;Wort -> wort; Wort
        (r'([a-z\xe4\xf6\xfc\xdf]);([A-Za-z\xc4\xd6\xdc\xe4\xf6\xfc])', r'\1; \2',
         'Fehlender Abstand nach Semikolon'),
        # Gesprochene Füllwörter entfernen (Whisper-Artefakte aus Podcasts)
        (r'(?i)\s*\b(?:ähm+|äh|öh|hmm?|mhm|ehm)\b\s*', ' ',
         'Füllwort ersetzen (ähm/äh → leer)'),
        # UTF-8 Bytes als ISO-8859-16 (Rumänisch) oder CP1250 falsch dekodiert
        # Erkennbar an Ă (U+0102) gefolgt von einem nicht-ASCII Zeichen
        # Reparatur: .encode(encoding).decode('utf-8')
        (r'Ă[^\x00-\x7f]',
         lambda m: next(
             (m.group().encode(enc).decode('utf-8')
              for enc in ('iso-8859-16', 'cp1250')
              if _try_encode(m.group(), enc)),
             m.group()
         ),
         'ISO-8859-16/CP1250 Mojibake (Ă + Sonderzeichen → Umlaut)'),
        (r'  +', ' ', 'Doppeltes Leerzeichen'),
        (r',\s*,', ',', 'Doppeltes Komma nach Füllwort'),
    ]
    return [(re.compile(p, re.UNICODE), repl, desc) for p, repl, desc in pairs]

REPLACEMENTS = _build_replacements()

MIN_WORDS = 4   # Zeilen mit weniger Woertern nach allen Replacements verwerfen
MAX_WORDS = 60  # Podcast-Run-ons: Sätze über 60 Wörter raus

# ── Quelldateien ──────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/tatoeba_de.txt"),
    Path("data/c4_de.txt"),
    *sorted(Path("data").glob("synthetic_*.txt")),
    Path("data/parlamentsrevue_de.txt"),
    Path("data/lnp_de.txt"),
    Path("data/minkorrekt_de.txt"),
    Path("data/raumzeit_de.txt"),
    Path("data/forschergeist_de.txt"),
    Path("data/cre_de.txt"),
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
                        help="Datei mit S\xe4tzen hoher Perplexity (eine pro Zeile); "
                             "exakte \xdcbereinstimmungen werden entfernt")
    args = parser.parse_args()

    patterns = []
    for entry in BANNED:
        p, desc = entry[0], entry[1]
        flags = entry[2] if len(entry) > 2 else re.IGNORECASE
        patterns.append((re.compile(p, flags), desc))

    # Optionale Menge von Sätzen aus --high-ppl
    high_ppl_set: set[str] = set()
    if args.high_ppl:
        ppl_path = Path(args.high_ppl)
        if not ppl_path.exists():
            print(f"Fehler: --high-ppl Datei nicht gefunden: {ppl_path}", file=sys.stderr)
            sys.exit(1)
        high_ppl_set = {l.strip() for l in ppl_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"{len(high_ppl_set)} S\xe4tze aus {ppl_path} geladen (hohe Perplexity).")

    sources = [Path(f) for f in args.files] if args.files else SOURCES
    seen_hashes: set[int] = set()
    total_removed = 0

    for src in sources:
        if not src.exists():
            continue

        tmp = src.with_suffix(".tmp")
        removed_counts = {entry[1]: 0 for entry in BANNED}
        n_ppl_removed = 0
        n_kept = 0
        n_total = 0
        n_dedup_removed = 0
        n_short_removed = 0
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
                    line = line.strip()
                    if line:
                        line = line[0].upper() + line[1:]
                    words = len(line.split())
                    if words < MIN_WORDS or words > MAX_WORDS:
                        n_short_removed += 1
                    else:
                        fout.write(line + "\n")
                        n_kept += 1

                if n_total % 5_000_000 == 0:
                    print(f"  {src.name}: {n_total:,} gelesen …", flush=True)

        n_removed = n_total - n_kept
        total_removed += n_removed

        if n_removed == 0:
            print(f"{src}: keine Treffer ({n_total:,} S\xe4tze)")
            if not args.dry_run:
                tmp.unlink()
            continue

        print(f"{src}: {n_removed:,} S\xe4tze entfernt (von {n_total:,})")
        if n_short_removed:
            print(f"  {n_short_removed:>8,}\xd7  zu kurz (< {MIN_WORDS} Woerter nach Cleaning)")
        if n_ppl_removed:
            print(f"  {n_ppl_removed:>8,}\xd7  hohe Perplexity (--high-ppl)")
        if n_dedup_removed:
            print(f"  {n_dedup_removed:>8,}\xd7  Duplikat (--dedup)")
        for desc, count in removed_counts.items():
            if count:
                print(f"  {count:>8,}\xd7  {desc}")

        for desc, count in replacement_counts.items():
            if count:
                print(f"  {count:>8,}\xd7  {desc}")

        if not args.dry_run:
            tmp.replace(src)

    print(f"\nGesamt entfernt: {total_removed:,}")
    if args.dry_run:
        print("(Dry-run — keine Datei ver\xe4ndert)")


if __name__ == "__main__":
    main()
