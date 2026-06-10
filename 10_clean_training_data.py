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
  .venv_ml/bin/python 10_clean_training_data.py --resume
  .venv_ml/bin/python 10_clean_training_data.py --pause   # aus zweitem Terminal
"""

import argparse
import hashlib
import html
import json
import multiprocessing as mp
import os
import re
import signal
import sys
from pathlib import Path

# ── Unerwünschte Muster ───────────────────────────────────────────────────────
# Jeder Eintrag: (pattern, beschreibung[, flags])
# Standard-Flags: re.IGNORECASE. Drittes Element = explizite Flags (z.B. 0).

# Sprach-unabhängige Web-Artefakte: Spam, Struktur, Format-Müll
WEB_ARTIFACTS = [
    # Tabellenzeilen, Listings
    (r"(?:[^|\t\n]*(?:\||\t)){2,}",  "Tabellenzeile (≥2 Pipes/Tabs) — kein Flie\xdftext"),
    (r"(?:[^:]*:){3}",      "≥3 Doppelpunkte (Produkt-/Listing-Spam)"),
    (r"[*•\xb7]{4,}",  "Bullet-Spam (≥4 Sonderzeichen in Folge)"),

    # Schlechte Satzenden
    (r"\b[a-z][a-z0-9-]{3,}\.[a-z]{2,6}\s*$",
     "Domain/URL am Satzende (managerfragen.org, loewenvergleich.de \u2026)"),
    (r"(?:\.{2,}|\u2026)\s*$", "Satzende mit .. / ... / … (unvollständig/abgebrochen)"),
    (r":\s*$",                    "Satzende mit Doppelpunkt (Überschrift/Intro-Fragment)"),
    (r"\[\.\.\.|\[…\]", "Eckige Klammer mit Auslassungspunkten ([...]/[…])"),
    (r"[!?]{2,}\s*$",       "Satzende mit !! oder ?? (Ausrufe-Spam)"),
    (r"\s(?:Dr|Prof|Hr|Fr|Nr|Col|Gen|Lt|Cpt|St|Tel|Bd|Mio|Mrd|Jh|bzw|ggf|inkl|usw|vgl|vs|Abb|Abs|ca|max|min|Fam|o\.\s*g|o\.\s*[äa]|u\.\s*[äa]|u\.\s*a|z\.\s*B|d\.\s*h|i\.\s*d\.\s*R|gem)\.\s*$",
     "Satzende mit Abkürzung (abgeschnittener Satz)"),
    (r"\s(?:[1-9]|[12]\d|3[01])\.\s*$",
     "Satzende mit Tagesdatum (abgeschnittener Satz: vom 22., bis zum 7.)"),

    # Strukturelle Artefakte
    (r"^-\s+\S",            "Zeile beginnt mit Listenpunkt (- item)"),
    (r"^[•▪▸►*|]",            "Zeile beginnt mit Aufzählungszeichen/Pipe (•▪▸►*|)"),
    (r"^[a-z][)]\s",        "Zeile beginnt mit Listenpräfix (a) item, b) item)"),
    (r"^\[",                 "Zeile beginnt mit [ (Blog-Tag, Kategorie-Header)"),
    (r"(?:.*#){3}",          "≥3 Hashtags (Social-Media-Tag-Spam)"),
    (r":\s*:",               "Doppelter Doppelpunkt (Formular-/Template-Artefakt)"),
    (r"\bN\xe4chste\s+Seite\b", "Paginierungs-Navigation (Nächste Seite)"),
    (r"\?[a-zäöüß]{2}",     "Fragezeichen statt ß (Kodierungsfehler: Bekannterma?en)"),
    (r"^(?:(?![äöüÄÖÜß]).)*\b(?:the|find|visit|recent|your|with|from|that|this|have|will|are|were|been|their|they|what|which|when|where|how|you|our|more|also|most|some|all|can|not|for|its|but|and|has|was|his|her|of)\b(?:(?![äöüÄÖÜß]).)*$",
     "Englischer Satz (keine Umlaute + englische Funktionswörter)", re.IGNORECASE),
    # "by" als Einzelwort (lowercase) ist kein deutsches Wort
    (r"\bby\b", "Englisches 'by' als Einzelwort (nie deutsches Wort)", 0),
    # Blog-Zeitstempel: "vor 3 Stunden / vor 2 Tagen" — Web-Metadaten
    # Nur Stunden/Minuten/Tage — "vor 100 Jahren" ist legitimer Fließtext
    (r"\bvor\s+\d+\s+(?:Stunden?|Minuten?|Tagen?)\b",
     "Blog-Zeitstempel (vor 3 Stunden / vor 2 Tagen …)"),
    # Programm-Listings: mehrere Uhrzeiten in einem Satz (Event-Programme, Stundenpläne)
    (r"\d{1,2}[:.]\d{2}\s*Uhr.{3,80}\d{1,2}[:.]\d{2}\s*Uhr",
     "Programm-Listing (mehrere Uhrzeiten im Satz — Event-Programm/Stundenplan)"),
    # Leere Klammern — UI-Fragmente ("(davon )", "(0)")
    (r"\(\s*\)",
     "Leere Klammer — UI-Fragment ((davon ), (0))"),
    # Englische Einschübe in deutschen Sätzen: Wörter die nie auf Deutsch vorkommen
    (r"\b(?:primarily|consists|available|therefore|however|although|whether|without"
     r"|another|between|because|something|nothing|everything|anything|already|usually"
     r"|actually|simply|previously|currently|especially|particularly|generally|basically)\b",
     "Englischer Einschub in deutschem Satz (primarily/consists/available …)", re.IGNORECASE),
    (r"(?:[^>]*>){2,}",      "≥2 > (Breadcrumb-Navigation)"),
    (r"\u2192|\u279c|\u27a1|\u203a|\u2039", "HTML-Navigationspfeil/Breadcrumb (→ ➜ ➡ › ‹ in Linktext)"),
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
    # Umfassende Unicode-Emoji-Ranges:
    #   U+1F000-U+1FAFF  moderne Emoji (Gesichter, Objekte, Flaggen …)
    #   U+2300-U+27BF    Misc Technical + Misc Symbols + Dingbats (☀⏯✂ …)
    #   U+2900-U+2BFF    Suppl. Arrows + Misc Math + Misc Symbols and Arrows
    #   U+FE00-U+FE0F    Variation Selectors (Emoji-Modifier)
    ("[\U0001F000-\U0001FAFF\U00002300-\U000027BF\U00002900-\U00002BFF\U0000FE00-\U0000FE0F]",
     "Unicode-Emoji / Symbol (umfassend: Faces, Objekte, Pfeile, Dingbats …)"),

    # Fotokredit (strukturell, nicht sprachspezifisch)
    (r"FOTOS?:",              "Fotokredit (FOTO:, FOTOS: ...)"),

    # All-Caps-Sätze (Forum-Header, Werbebanner, Clickbait-Titel)
    (r"^[A-Z\xc4\xd6\xdc][A-Z\xc4\xd6\xdc\s\-!?.,:0-9]{15,}$",
     "All-Caps-Satz (Werbebanner/Forum-Header)", 0),

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
    (r"\bHalo\b",       "Halo (echtes Wort, aber kein sinnvoller Tippvorschlag)", 0),

    # Literaturverweise: (Hg.) / (Hrsg.) — Zitationsformat, kein Fließtext
    (r"\(Hg\.\)|\(Hrsg\.\)",
     "Literaturverweis (Hg./Hrsg. — Zitationsformat)"),
    # Clickbait-Listicle-Titel: "Diese 5 Dinge/Tipps/Wege musst du …"
    (r"\bDiese\s+\d+\s+(?:Dinge|Tipps?|Wege?|Gr\xfcnde?|Fakten|Tricks?|Methoden?|Schritte?|Punkte?|Fragen?|Fehler|Zeichen)\b",
     "Clickbait-Listicle (Diese 5 Dinge / 10 Tipps …)", 0),

    # Römische Zahlen — Einzelbuchstaben I/V/X/L/C/D/M alle ausgeschlossen
    # (Kollision mit Abkürzungen, Vitamin C, USB-C, etc.)
    # CD ausgeschlossen wegen Compact Disc; nur Mehrzeichenkombinationen
    (r"\b(?:"
     r"M{2,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"  # ≥2000
     r"|M(?:CM|CD|D?C{1,3}|(?:XC|XL|L?X{1,3})|IX|IV|VIII|VII|VI|III|II)"  # 1000+Rest (mind. 2 Zeichen)
     r"|CM|DM|DC{1,3}|C{2,3}M?|C{1,3}(?:XC|XL|L?X{1,3})|C{1,3}(?:IX|IV|VIII|VII|VI|III|II)"  # 100–899 mehrzeichig
     r"|(?:XC|XL|L?X{1,3})(?:IX|IV|V?I{1,3})|XC|XL|XX{1,2}|LX{0,3}I{1,3}|LX{1,3}"  # 10–99 mehrzeichig
     r"|IX|IV|VIII|VII|VI|III|II"                                          # 2–9 explizit
     r")\b",
     "Römische Zahl (II–MMMCMXCIX — Einzelbuchstaben C/D/M/X/L/V/I alle ausgeschlossen)", 0),

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
    (r"\d+,-\s*€",        "Preis-Listing (149,- € — Komma-Strich-Notation)"),
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
    (r"\b(?:online|internet)\s+casinos?\b",
     "Casino-SEO-Injektion (online/internet casino(s) in normalem Satz)"),

    # Adult-Content-SEO (Escort-Seiten-Spam in FineWeb2)
    (r"\bsexy\s+frau\b|\bnudisten\b|\bdominante\s+massagen\b"
     r"|\bFicken\b|\bMilf\b|\bfürsex\b|\bBipaar\b|\bprostituiert\w*\b",
     "Adult-SEO-Spam (sexy frau / nudisten / Ficken / Milf …)"),
    (r"porn",                "Pornografische URL/Begriff (porn als Substring)"),

    # Wochentags-Abkürzungen (Stundenpläne, Öffnungszeiten — kein Fließtext)
    (r"\bMo\b|\bDi\b|\bMi\b|\bDo\b|\bFr\b|\bSa\b",
     "Wochentags-Abkürzung (Mo/Di/Mi/Do/Fr/Sa — Stundenpläne/Öffnungszeiten)", 0),

    # Korrumpierte/obskure Wörter und Abkürzungen
    (r"\bhapre\b",  "Korrumpiertes Wort (hapre)", re.IGNORECASE),
    (r"\boll\b",    "Fragment/Dialekt (oll)", re.IGNORECASE),
    (r"\bwa\b",     "Fragment/Dialekt (wa)", re.IGNORECASE),
    (r"\bHd\b",     "Malformierte Abkürzung (Hd statt HD)", 0),
    (r"\bTWAs?\b",  "Obskure Abkürzung (TWA/TWAs)", 0),
    (r"\bEWS\b",    "Obskure Abkürzung (EWS)", 0),
    (r"\bLSKen?\b", "Obskure Abkürzung (LSK/LSKen)", 0),
    (r"\bMDK\b",    "Obskure Abkürzung (MDK)", 0),
    (r"\bNDP\b",    "Obskure Abkürzung (NDP)", 0),
    (r"\bMdL\b",    "Parlamentsabkürzung (MdL — Mitglied des Landtags)", 0),
    (r"\bAich\b",         "Hyperlokal-Ortsname (Aich)", 0),
    (r"\bWeeze\b",        "Hyperlokal-Ortsname (Weeze)", 0),
    (r"\bSchwedt\b",      "Hyperlokal-Ortsname (Schwedt)", 0),
    (r"\bMeckenheims?\b",          "Hyperlokal-Ortsname (Meckenheim/Meckenheims)", 0),
    (r"\bAalenerInnen?\b",         "Hyperlokal + Genderschreibung (AalenerIn/Innen)", 0),
    (r"\bAalenerins?\b",           "Hyperlokal-Einwohnerin (Aalenerin)", 0),
    (r"\bMediendom\b",             "Obskurer Eigenname (Mediendom)", 0),
    (r"\bBanaters?\b",             "Obskurer Begriff (Banater/Banaters)", 0),
    (r"\bFonems?\b",               "Obskurer/fehlerhafter Begriff (Fonem statt Phonem)", 0),
    (r"\bSSe\b",                   "Korrumpierte Abkürzung/Artefakt (SSe)", 0),
    (r"\bLeu\b",                   "Rumänische Währung (Leu — kein deutsches Wort)", 0),
    (r"\bChet\b",                  "Fremdsprachiger Eigenname (Chet)", 0),
    (r"\bj-ja\b",                  "Stotter-Artefakt (j-ja)", 0),
    (r"\bFibu\b",                  "Obskure Abkürzung (Fibu — Finanzbuchhaltung)", 0),
    (r"\bFine\b",         "Englisches/ital. Fremdwort (Fine)", 0),

    # E-Mail-Abschlussformel
    (r"\bMit freundlichen Gr\xfc\xdfen\b",
     "E-Mail-Abschlussformel (Mit freundlichen Grüßen)"),

    # Kommentarbox-UI
    (r"\d+\s*Zeichen\s*(?:übrig|verbleibend)", "Zeichenzähler (Web-Kommentarbox-UI)"),
    (r"\bSchreiben Sie.*Kommentar\b", "Kommentarbox-Aufforderung"),
    (r"\bmehr\s+lesen\b|\bweiterlesen\b|[.!?]\s+Mehr\s*$",
     "Web-Artefakt (mehr lesen / weiterlesen / ». Mehr« — Teaserlink)"),

    # E-Commerce-Header
    (r"\bBestellen oder Ausleihen\b", "E-Commerce-Header (Kaufen, Bestellen oder Ausleihen)"),
]

# Nicht-deutsche Schriften und Zeichen (Sprach-Ausschlüsse)
LANG_EXCLUDE = [
    # Nicht-deutsche europäische Sonderzeichen: Estnisch/Finnisch/Polnisch etc.
    (r"[õőűčšžāēīūţşąęśźżłńçğı]",
     "Nicht-dt. EU-Buchstabe (\xf5čšžāłçğı — Estnisch/Polnisch/Lettisch/T\xfcrkisch …)", 0),
    # Fremdsprachige Ortsnamen/Wörter die nie in deutschen Sätzen vorkommen sollten
    (r"\bFünens?\b", "Dänischer Ortsname (Fünen/Fünens — dän. Insel Fyn)", 0),
    (r"\bÖr\b",      "Schwedisch/Dänisch (Ör — kein deutsches Wort)", 0),

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
        # Mojibake: UTF-8-Bytes als Latin-1 gelesen → Umlaute reparieren
        # Muster: Ã (U+00C3) + kombinierender Buchstabe statt korrektem Umlaut
        ("\xc3\xa4", "\xe4", "Mojibake ä (Ã¤ → ä)"),
        ("\xc3\xb6", "\xf6", "Mojibake ö (Ã¶ → ö)"),
        ("\xc3\xbc", "\xfc", "Mojibake ü (Ã¼ → ü)"),
        ("\xc3\x84", "\xc4", "Mojibake Ä (Ã\x84 → Ä)"),
        ("\xc3\x96", "\xd6", "Mojibake Ö (Ã\x96 → Ö)"),
        ("\xc3\x9c", "\xdc", "Mojibake Ü (Ã\x9c → Ü)"),
        ("\xc3\x9f", "\xdf", "Mojibake ß (Ã\x9f → ß)"),
        # C1-Steuerzeichen (U+0080-U+009F): Windows-1252-Artefakte
        ("[\x80-\x9f]", "", "C1-Steuerzeichen entfernen (Windows-1252-Artefakte)"),
        # Mojibake Typ 2: UTF-8-Bytes als Windows-1252 gelesen (ae-Euro-Sequenzen)
        ("\xe2\x80\x9e", "",  "Mojibake ae-Euro-z -> Anf.-Zeichen (wird gestrichen)"),
        ("\xe2\x80\x9c", "",  "Mojibake ae-Euro-oe -> Anf.-Zeichen (wird gestrichen)"),
        ("\xe2\x80\x93", "-",  "Mojibake ae-Euro-lq -> Gedankenstrich"),
        ("\xe2\x80\x91", "'", "Mojibake ae-Euro-rq -> Apostroph"),
        ("\xe2\x80\x99", "",  "Mojibake ae-Euro-tm -> Anf.-Zeichen (wird gestrichen)"),
        # Anführungszeichen aller Art entfernen
        # U+0022 gerade, U+00AB/BB Guillemets, U+2018-201F alle typograf. Einzel-/Doppelanführungen
        (r'["«»‘-‟]', "",  'Anführungszeichen entfernen (", «», ‘-‟)'),
        # Zitatmarker und Pfeilzeichen entfernen: > Zitat, <Verweis>
        (r'[<>]', '', 'Spitze Klammern (Zitatmarker, Pfeil)'),
        # Fehlende Leerzeichen nach Satzzeichen reparieren
        # Weiches Trennzeichen (U+00AD): Formatierungsartefakt aus Web-Quellen
        ("\xad", "", "Weiches Trennzeichen (U+00AD) entfernen"),
        # Häufige deutsche Abkürzungen ausschreiben — VOR dem Punkt-Spacing,
        # damit z.B. nicht zu "z. B." aufgetrennt wird (→ Single-Letter-Tokens)
        (r'\bz\.B\.', 'zum Beispiel', 'z.B. → zum Beispiel'),
        (r'\bz\.b\.', 'zum Beispiel', 'z.b. → zum Beispiel'),
        (r'\bd\.h\.', 'das heißt',    'd.h. → das heißt'),
        (r'\bu\.a\.', 'unter anderem', 'u.a. → unter anderem'),
        (r'\bu\.U\.', 'unter Umständen', 'u.U. → unter Umständen'),
        (r'\bz\.T\.', 'zum Teil',      'z.T. → zum Teil'),
        (r'\bv\.a\.', 'vor allem',     'v.a. → vor allem'),
        (r'\bi\.d\.R\.', 'in der Regel', 'i.d.R. → in der Regel'),
        (r'\bi\.A\.', 'im Auftrag',    'i.A. → im Auftrag'),
        (r'\bs\.o\.', 'siehe oben',    's.o. → siehe oben'),
        (r'\bs\.u\.', 'siehe unten',   's.u. → siehe unten'),
        (r'\ba\.a\.O\.', 'am angeführten Ort', 'a.a.O. → am angeführten Ort'),
        (r'\bo\.ä\.', 'oder ähnliches', 'o.ä. → oder ähnliches'),
        (r'\bo\.Ä\.', 'oder Ähnliches', 'o.Ä. → oder Ähnliches'),
        (r'\bggf\.', 'gegebenenfalls', 'ggf. → gegebenenfalls'),
        (r'\bbzw\.', 'beziehungsweise', 'bzw. → beziehungsweise'),
        (r'\busw\.', 'und so weiter',  'usw. → und so weiter'),
        (r'\binkl\.', 'inklusive',     'inkl. → inklusive'),
        (r'\bca\.', 'circa',           'ca. → circa'),
        (r'\bevtl\.', 'eventuell',     'evtl. → eventuell'),
        (r'\bvgl\.', 'vergleiche',     'vgl. → vergleiche'),
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
        # Auslassungspunkte im Satzinneren (... / …) → Leerzeichen
        # Satzenden mit ... werden bereits durch BAN_PATTERNS gebannt;
        # hier werden die mittleren Vorkommen bereinigt damit das Modell
        # keine Auslassungspunkte als Vorschläge lernt.
        (r'(?:\.{2,}|…)', ' ', 'Auslassungspunkte im Satz (... / … → Leerzeichen)'),
        (r'  +', ' ', 'Doppeltes Leerzeichen'),
        (r',\s*,', ',', 'Doppeltes Komma nach Füllwort'),
    ]
    return [(re.compile(p, re.UNICODE), repl, desc) for p, repl, desc in pairs]

REPLACEMENTS = _build_replacements()

MIN_WORDS = 4   # Zeilen mit weniger Woertern nach allen Replacements verwerfen
MAX_WORDS = 60  # Podcast-Run-ons: Sätze über 60 Wörter raus

CHECKPOINT_FILE = Path(__file__).parent / ".10_clean_checkpoint.json"
PAUSE_FILE      = Path(__file__).parent / ".10_clean_pause"

CHUNK_SIZE = 200_000  # Zeilen pro Worker-Chunk

# Einmal kompilierte Pattern-Liste — via fork von Worker-Prozessen geerbt
PATTERNS: list = []
for _e in BANNED:
    _flags = _e[2] if len(_e) > 2 else re.IGNORECASE
    PATTERNS.append((re.compile(_e[0], _flags), _e[1]))

# Worker-State (wird per Pool-Initializer gesetzt)
_W_HIGH_PPL: frozenset = frozenset()


def _worker_init(high_ppl_set: frozenset) -> None:
    global _W_HIGH_PPL
    _W_HIGH_PPL = high_ppl_set


def _process_chunk(lines: list[str]):
    kept = []
    removed: dict[str, int] = {}
    replaced = {desc: 0 for _, _, desc in REPLACEMENTS}
    n_ppl = n_short = 0
    for line in lines:
        stripped = line.strip()
        if stripped in _W_HIGH_PPL:
            n_ppl += 1
            continue
        drop = False
        for pat, desc in PATTERNS:
            if pat.search(line):
                removed[desc] = removed.get(desc, 0) + 1
                drop = True
                break
        if not drop:
            for pat, repl, desc in REPLACEMENTS:
                new_line, n = pat.subn(repl, line)
                if n:
                    replaced[desc] += n
                    line = new_line
            line = line.strip()
            if line:
                line = line[0].upper() + line[1:]
            words = len(line.split())
            if words < MIN_WORDS or words > MAX_WORDS:
                n_short += 1
            else:
                kept.append(line)
    return kept, removed, replaced, n_ppl, n_short, len(lines)


def _read_chunks(fh, size: int):
    chunk: list[str] = []
    for line in fh:
        chunk.append(line)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# ── Quelldateien ──────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/tatoeba_de.txt"),
    Path("data/c4_de.txt"),
    Path("data/fineweb2_de.txt"),
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
    parser.add_argument("--pause", action="store_true",
                        help="Laufendes Skript nach aktueller Datei anhalten (Pause-Anforderung setzen)")
    parser.add_argument("--resume", action="store_true",
                        help="Verarbeitung ab dem letzten Checkpoint fortsetzen")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2),
                        help=f"Parallele Worker (default: CPU-Kerne minus 2 = {max(1, mp.cpu_count() - 2)})")
    args = parser.parse_args()

    # --pause: Pause-Signal für laufende Instanz setzen, dann beenden
    if args.pause:
        PAUSE_FILE.touch()
        print("Pause-Anforderung gesetzt — das laufende Skript hält nach der aktuellen Datei an.")
        print(f"Fortsetzen mit: --resume")
        return

    # --resume: bereits verarbeitete Dateien aus Checkpoint laden
    done_files: set[str] = set()
    if args.resume:
        if CHECKPOINT_FILE.exists():
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            done_files = set(data.get("done", []))
            print(f"Fortsetzen: {len(done_files)} Datei(en) bereits verarbeitet — werden übersprungen.")
        else:
            print("Kein Checkpoint gefunden — starte von vorne.")
        PAUSE_FILE.unlink(missing_ok=True)

    # Ctrl+C: beim ersten Mal graceful pause, beim zweiten Mal sofort abbrechen
    pause_flag = [False]

    def _on_sigint(sig, frame):
        if pause_flag[0]:
            print("\nZweites Signal — sofortiger Abbruch.", file=sys.stderr)
            sys.exit(130)
        pause_flag[0] = True
        print("\nCtrl+C: Pause nach aktueller Datei … (nochmals Ctrl+C zum sofortigen Abbruch)",
              file=sys.stderr)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

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
    paused = False

    for src in sources:
        if not src.exists():
            continue

        if str(src) in done_files:
            print(f"{src}: übersprungen (Checkpoint)")
            continue

        tmp = src.with_suffix(".tmp")
        removed_counts = {entry[1]: 0 for entry in BANNED}
        n_ppl_removed = 0
        n_kept = 0
        n_total = 0
        n_dedup_removed = 0
        n_short_removed = 0
        replacement_counts = {desc: 0 for _, _, desc in REPLACEMENTS}

        n_workers = args.workers
        if args.dedup and n_workers > 1:
            print(f"  Hinweis: --dedup erfordert seriellen Modus (--workers 1).")
            n_workers = 1

        with src.open("r", encoding="utf-8") as fin, \
             (tmp.open("w", encoding="utf-8") if not args.dry_run else open(os.devnull, "w")) as fout:

            if n_workers > 1:
                with mp.Pool(n_workers, initializer=_worker_init,
                             initargs=(frozenset(high_ppl_set),)) as pool:
                    for r_kept, r_removed, r_replaced, r_ppl, r_short, r_n in \
                            pool.imap(_process_chunk, _read_chunks(fin, CHUNK_SIZE), chunksize=2):
                        prev = n_total // 5_000_000
                        n_total += r_n
                        n_kept += len(r_kept)
                        n_ppl_removed += r_ppl
                        n_short_removed += r_short
                        for desc, count in r_removed.items():
                            removed_counts[desc] += count
                        for desc, count in r_replaced.items():
                            replacement_counts[desc] += count
                        for line in r_kept:
                            fout.write(line + "\n")
                        if n_total // 5_000_000 > prev:
                            print(f"  {src.name}: {n_total:,} gelesen …", flush=True)
            else:
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
                    for pat, desc in PATTERNS:
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
        else:
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

        # Checkpoint nach jeder Datei aktualisieren (auch bei 0 Treffern)
        if not args.dry_run:
            done_files.add(str(src))
            CHECKPOINT_FILE.write_text(
                json.dumps({"done": sorted(done_files)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Pause-Anforderung prüfen (--pause aus zweitem Terminal oder Ctrl+C)
        if pause_flag[0] or PAUSE_FILE.exists():
            PAUSE_FILE.unlink(missing_ok=True)
            paused = True
            print("\nAngehalten. Fortsetzen mit: --resume")
            break

    print(f"\nGesamt entfernt: {total_removed:,}")
    if args.dry_run:
        print("(Dry-run — keine Datei ver\xe4ndert)")
    elif not paused and not args.dry_run:
        # Alles fertig — Checkpoint aufräumen
        CHECKPOINT_FILE.unlink(missing_ok=True)
        print("Checkpoint gelöscht (alle Dateien vollständig verarbeitet).")


if __name__ == "__main__":
    main()
