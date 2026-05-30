#!/usr/bin/env python3
"""
Generiert themenbasierte synthetische Trainingssätze für unterrepräsentierte Domänen.

Im Gegensatz zu 08_generate_synthetic.py (generischer Handy-Stil) zielt dieses
Skript auf spezifische Themengebiete mit maßgeschneiderten Prompts — Geschichte,
Medizin, Technik, Natur usw. Jedes Thema hat einen eigenen Kontext und Stil.

Output: data/synthetic_themen.txt  (separater File, nicht synthetic_de.txt)

Usage:
  .venv_ml/bin/python 12_generate_synthetic_vocab.py [--per-topic 50]
  .venv_ml/bin/python 12_generate_synthetic_vocab.py --topics mittelalter,medizin
  .venv_ml/bin/python 12_generate_synthetic_vocab.py --list-topics
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")

# ── Themen mit spezifischen Prompts ──────────────────────────────────────────
# Jedes Thema: name, beschreibung, system_zusatz, beispiele
# system_zusatz ergänzt den allgemeinen System-Prompt für das Thema

TOPICS = {
    "mittelalter": {
        "name": "Mittelalter & Geschichte",
        "beschreibung": (
            "Sätze rund ums Mittelalter und mittelalterliche Geschichte — "
            "so wie man sie in einem Gespräch, Chat oder beim Googeln tippt. "
            "Nicht Wikipedia-Stil, sondern wie echte Menschen darüber sprechen."
        ),
        "beispiele": [
            "Weißt du noch, was im Mittelalter mit Ketzern passiert ist?",
            "Die Burg Hohenzollern ist aus dem Mittelalter, oder?",
            "Ich finde das Mittelalter total faszinierend.",
            "Haben die Ritter im Mittelalter wirklich so gelebt?",
            "Das erinnert mich irgendwie ans Mittelalter.",
            "Wir waren gestern auf dem Mittelaltermarkt, war super!",
            "Karl der Große gilt ja als Vater Europas.",
            "Die Pest hat im Mittelalter Millionen Menschen getötet.",
        ],
    },
    "geschichte": {
        "name": "Deutsche Geschichte",
        "beschreibung": (
            "Sätze zur deutschen Geschichte — Kaiserreich, Weimar, NS-Zeit, DDR, "
            "Wiedervereinigung, BRD. Alltägliche Gesprächskontexte: Schule, Nachrichten, "
            "Dokumentationen, Familiengeschichten, Museumsbesuche."
        ),
        "beispiele": [
            "Habt ihr in Geschichte gerade die Weimarer Republik?",
            "Der Mauerfall war echt ein historischer Moment.",
            "Mein Opa hat noch den Zweiten Weltkrieg erlebt.",
            "Das Bundestag-Gebäude hat so eine bewegte Geschichte.",
            "Ich finde die Nachkriegszeit total spannend.",
            "Die Teilung Deutschlands hat so viele Familien zerrissen.",
            "Bismarck hat das Deutsche Reich ja eigentlich erst gegründet.",
            "Die Novemberrevolution 1918 hab ich nie richtig verstanden.",
        ],
    },
    "medizin": {
        "name": "Medizin & Gesundheit",
        "beschreibung": (
            "Sätze zu Gesundheitsthemen, Arztbesuchen, Diagnosen, Behandlungen — "
            "so wie man sie im Alltag tippt. Gespräche mit Freunden, Familie, "
            "Terminvereinbarungen, Erklärerungen, Fragen."
        ),
        "beispiele": [
            "Ich muss heute zum Orthopäden wegen meinem Rücken.",
            "Die Entzündungswerte waren leider erhöht.",
            "Mein Hausarzt hat mich zum Kardiologen überwiesen.",
            "Bluthochdruck ist in unserer Familie leider genetisch.",
            "Hast du eine gute Empfehlung für einen Neurologen?",
            "Das MRT hat nichts Schlimmes gezeigt, Gott sei Dank.",
            "Ich nehme jetzt Metformin gegen den Diabetes.",
            "Die Physiotherapie hilft echt, mein Knie wird besser.",
        ],
    },
    "computer": {
        "name": "Computer & Technologie",
        "beschreibung": (
            "Sätze rund um Computer, Software, Internet, KI, Gaming — "
            "Alltagsgespräche über Technik, Fragen, Empfehlungen, Probleme. "
            "Nicht nur Fachjargon, auch wie Normalnutzer über Technik sprechen."
        ),
        "beispiele": [
            "Mein Laptop ist schon wieder abgestürzt.",
            "Hast du das neue ChatGPT-Update schon ausprobiert?",
            "Ich versteh das Konzept hinter Blockchain immer noch nicht.",
            "Linux ist für mich persönlich einfach zu kompliziert.",
            "Welche GPU würdest du für Gaming empfehlen?",
            "Die Cloud-Synchronisation macht mir Sorgen wegen Datenschutz.",
            "Das neue iPhone hat angeblich einen besseren Chip.",
            "Ich lern gerade Python, macht echt Spaß.",
        ],
    },
    "natur": {
        "name": "Natur & Umwelt",
        "beschreibung": (
            "Sätze zu Natur, Tieren, Pflanzen, Umweltschutz, Klimawandel — "
            "wie man im Alltag darüber spricht: beim Wandern, in Nachrichten, "
            "Gesprächen über Haustiere, den Garten, das Wetter."
        ),
        "beispiele": [
            "Wir haben heute einen Rotmilan gesehen beim Wandern!",
            "Die Eiche in unserem Garten ist wirklich uralt.",
            "Die Schmetterlingspopulationen gehen dramatisch zurück.",
            "Hast du die Doku über die Korallenriffe gesehen?",
            "Ich finde Kompostieren gar nicht so schwer wie gedacht.",
            "Die Hitzeperioden werden immer schlimmer wegen Klimawandel.",
            "Im Nationalpark darf man das Unterholz nicht berühren.",
            "Dieser Sommer ist viel zu trocken für die Landwirtschaft.",
        ],
    },
    "politik": {
        "name": "Politik & Gesellschaft",
        "beschreibung": (
            "Sätze zu aktuellen politischen Themen, gesellschaftlichen Debatten, "
            "Wahlen, Parteien — wie man im Alltag, in Familiengesprächen oder "
            "mit Freunden darüber spricht. Keine Propaganda, normaler Diskurs."
        ),
        "beispiele": [
            "Ich versteh die Ampelkoalition manchmal wirklich nicht.",
            "Die Wahlbeteiligung war diesmal überraschend hoch.",
            "Hast du die Bundeskanzler-Rede gestern gesehen?",
            "Das Rentensystem muss dringend reformiert werden.",
            "Ich finde das neue Einwanderungsgesetz schwierig.",
            "Die EU macht manchmal echt sinnvolle Sachen.",
            "Klimapolitik ist wichtig, aber die Umsetzung lahmt.",
            "Der neue Bürgermeister hat interessante Ideen.",
        ],
    },
    "musik": {
        "name": "Musik & Konzerte",
        "beschreibung": (
            "Sätze zu Musik aller Genres, Konzerten, Festivals, Bands, Instrumenten — "
            "wie Musikfans im Alltag darüber schreiben. Empfehlungen, Konzerterlebnisse, "
            "Diskussionen über Alben, Künstler, Live-Auftritte."
        ),
        "beispiele": [
            "Das Konzert gestern Abend war absolut legendär.",
            "Rammstein live ist einfach ein anderes Erlebnis.",
            "Ich lern gerade Klavierspielen, bin totaler Anfänger.",
            "Das neue Album von Cro ist leider eher enttäuschend.",
            "Das Jazzfestival in Montreux steht auf meiner Bucketlist.",
            "Kammermusik klingt immer so elitär, aber ich find's schön.",
            "Habt ihr Tickets für das Open Air schon?",
            "Die Berliner Philharmoniker sind halt unschlagbar.",
        ],
    },
    "sport": {
        "name": "Sport & Bewegung",
        "beschreibung": (
            "Sätze zu Sport aller Art — Fußball, Laufen, Rad, Klettern, Schwimmen, "
            "Fitnessstudio, Mannschaftssport. Wie man über Training, Wettkämpfe, "
            "Vereine und Sportevents im Alltag spricht."
        ),
        "beispiele": [
            "Der Halbmarathon war brutal, aber ich hab's geschafft!",
            "Wann trainiert ihr nächste Woche?",
            "Mein Knie macht beim Laufen immer noch Probleme.",
            "Die Champions League gestern war enttäuschend.",
            "Ich habe jetzt einen Personal Trainer, lohnt sich echt.",
            "Beim Klettersteig braucht man unbedingt gutes Schuhwerk.",
            "Triathlon wäre mein nächstes großes Ziel.",
            "Das Stadion war voll, super Stimmung.",
        ],
    },
    "essen": {
        "name": "Essen & Kochen",
        "beschreibung": (
            "Sätze zu Rezepten, Restaurants, Kochprojekten, Ernährung — "
            "Alltagsgespräche über Essen: was jemand kocht, Restaurantempfehlungen, "
            "Ernährungsumstellungen, regionale Spezialitäten."
        ),
        "beispiele": [
            "Ich hab gestern Sauerbraten gemacht, erster Versuch.",
            "Kennst du ein gutes veganes Restaurant hier in der Nähe?",
            "Die Maultaschen vom Metzger sind wirklich authentisch.",
            "Ich probier gerade glutenfreie Ernährung aus.",
            "Das Rezept für Schwarzwälder Kirschtorte ist gar nicht so schwer.",
            "Laktoseintoleranz macht Käse kaufen echt kompliziert.",
            "Wir haben samstags immer Weißwurstfrühstück.",
            "Das Kölsch zum Schnitzel war perfekt.",
        ],
    },
    "medien": {
        "name": "Bücher, Musik & Film",
        "beschreibung": (
            "Sätze über Bücher, CDs, DVDs, Blu-rays, Hörbücher, Podcasts, Streaming — "
            "wie man im Alltag darüber spricht: Empfehlungen, Erlebnisse, Meinungen, "
            "Kauf- und Leihentscheidungen. Natürlicher Umgang mit Medien, kein Katalogstil."
        ),
        "beispiele": [
            "Ich hab das Buch letzte Woche ausgelesen, war wirklich gut.",
            "Hast du das schon als Hörbuch gehört? Ich find das praktischer.",
            "Die CD liegt bei mir noch irgendwo rum, hör ich kaum noch.",
            "Ich kaufe kaum noch DVDs, aber für alte Filme lohnt es sich.",
            "Das Buch hat mir jemand empfohlen, liegt jetzt seit Monaten ungelesen da.",
            "Die Blu-ray-Qualität ist bei dem Film wirklich beeindruckend.",
            "Ich höre lieber Podcasts als Musik beim Kochen.",
            "Das war das beste Buch, das ich seit Jahren gelesen habe.",
        ],
    },
    "haushalt": {
        "name": "Heimwerken & Haushalt",
        "beschreibung": (
            "Sätze rund ums Heimwerken, Reparieren und Einrichten — "
            "wie man im Alltag darüber spricht: Materialkauf, Werkzeug, "
            "Do-it-yourself-Projekte, Nachbarschaftshilfe, Tipps."
        ),
        "beispiele": [
            "Hast du noch Nägel da? Ich will das Bild aufhängen.",
            "Ich brauch längere Schrauben, die passen nicht.",
            "Der Dübel ist zu groß für das Loch.",
            "Kannst du mir kurz die Bohrmaschine leihen?",
            "Der Akkuschrauber von Bosch ist echt sein Geld wert.",
            "Ich muss noch das Regal an die Wand dübeln.",
            "Die Wasserleitung tropft, ich ruf morgen den Klempner.",
            "Weißt du ob M6 oder M8 Schrauben besser sind?",
        ],
    },
    "bad": {
        "name": "Badezimmer & Sanitär",
        "beschreibung": (
            "Sätze rund ums Badezimmer — Renovierung, Pflege, Einrichtung. "
            "Wie man über Badplanung, Reparaturen und Badprodukte im Alltag "
            "spricht: Gespräche mit Familie, Handwerkern, Freunden."
        ),
        "beispiele": [
            "Die Badewanne ist so alt, die muss irgendwann raus.",
            "Wir überlegen eine ebenerdige Dusche einzubauen.",
            "Die Silikonfugen in der Dusche sind leider schimmelig.",
            "Der Wasserhahn tropft schon seit Wochen.",
            "Neue Fliesen würden dem Bad wirklich gut tun.",
            "Wir wollen das Bad komplett renovieren lassen.",
            "Der Sanitär hat gesagt drei Wochen Wartezeit.",
            "Die Duschkabine ist wirklich mühsam zu reinigen.",
        ],
    },
    "wohnen": {
        "name": "Wohnen & Möbel",
        "beschreibung": (
            "Sätze rund um Einrichten, Möbel, Umzug und Wohngestaltung — "
            "Alltags-Gespräche über neue Möbel, Renovierung, Umzugsstress, "
            "IKEA-Abenteuer, Bodenbelag, Farben und Raumgefühl."
        ),
        "beispiele": [
            "Wir brauchen noch einen Kleiderschrank für das Schlafzimmer.",
            "Ich baue gerade eine IKEA-Kommode zusammen, nie wieder.",
            "Der Umzug war eine Katastrophe, drei Kartons kaputtgegangen.",
            "Hast du einen guten Tipp für günstigen Laminat?",
            "Das Sideboard passt perfekt an die Wand im Flur.",
            "Wir überlegen das Wohnzimmer neu zu streichen.",
            "Die Couch ist schon so durchgesessen, die muss weg.",
            "Das Regal aus massivem Holz sieht wirklich schön aus.",
        ],
    },
    "einkaufen": {
        "name": "Einkaufen & Supermarkt",
        "beschreibung": (
            "Sätze rund ums Einkaufen — Supermarkt, Wochenmarkt, Lebensmittel, "
            "Preise, Angebote. Wie man über den Alltags-Einkauf spricht: "
            "Einkaufslisten, Kassengespräche, Preisvergleiche, Marken."
        ),
        "beispiele": [
            "Ich muss noch schnell zum Supermarkt, hast du eine Liste?",
            "Die Einkaufsliste hab ich mal wieder vergessen.",
            "Hast du noch Milch geholt oder soll ich?",
            "Beim Wochenmarkt ist das Gemüse viel frischer.",
            "Die Schlangen an der Kasse samstags sind schrecklich.",
            "Ich versuche weniger Plastikverpackungen zu kaufen.",
            "Der Discounter ist günstiger aber das Bio-Gemüse fehlt.",
            "Wir kaufen seit kurzem öfter beim lokalen Bäcker statt im Supermarkt.",
        ],
    },
    "auto": {
        "name": "Auto & Werkstatt",
        "beschreibung": (
            "Sätze rund ums Auto — Werkstatt, TÜV, Reifenwechsel, Pannen, "
            "Kraftstoff, Pflege. Wie Autofahrer im Alltag über ihr Fahrzeug "
            "sprechen: Reparaturen, Kosten, Tipps, Ärger mit der Werkstatt."
        ),
        "beispiele": [
            "Mein Auto muss dringend zum TÜV, schon überfällig.",
            "Die Bremsscheiben müssen bald gewechselt werden.",
            "Der Winterreifen-Wechsel ist bei meiner Werkstatt schon gebucht.",
            "Die Werkstatt hat mal wieder viel zu viel verlangt.",
            "Kilometerstand über 200.000, aber läuft noch super.",
            "Hast du das Motoröl schon nachgefüllt?",
            "Die Klimaanlage macht komische Geräusche seit letztem Sommer.",
            "Ich wechsle die Felgen immer selbst, ist nicht schwer.",
        ],
    },
    "kleidung": {
        "name": "Kleidung & Mode",
        "beschreibung": (
            "Sätze rund um Kleidung, Mode und Shopping — Alltagsgespräche "
            "über Kauf, Passform, Qualität, Größen, Marken, Waschen und Stil. "
            "Kein Produktlisting-Stil, echter Austausch unter Menschen."
        ),
        "beispiele": [
            "Ich brauch noch eine Jacke für den Winter.",
            "Die Hose passt leider nicht mehr nach dem Urlaub.",
            "Welche Größe nimmst du normalerweise?",
            "Das T-Shirt ist nach dem ersten Waschen eingegangen.",
            "Ich kaufe Kleidung lieber in echt als online.",
            "Der Reisverschluss geht kaputt, typisch billige Qualität.",
            "Herbst ist meine liebste Jahreszeit wegen der Mode.",
            "Die Stiefel sind reduziert, ich überleg noch.",
        ],
    },
    "reise": {
        "name": "Reisen & Urlaub",
        "beschreibung": (
            "Sätze zu Reiseplänen, Urlaubserlebnissen, Sehenswürdigkeiten, "
            "Transportmitteln, Unterkünften — wie Reisende im Alltag schreiben: "
            "Tipps, Empfehlungen, Erlebnisberichte, Buchungsthemen."
        ),
        "beispiele": [
            "Habt ihr für den Städtetrip schon eine Unterkunft?",
            "Das Hostel in Lissabon war überraschend schön.",
            "Mit dem Fernbus nach München ist viel günstiger als die Bahn.",
            "Reisekrankenversicherung sollte man nie vergessen.",
            "Die Stadtführung in Prag war wirklich informativ.",
            "Wohnmobil-Urlaub wäre mal was anderes.",
            "Das Visum für Indien dauert ewig.",
            "Der Museumspass lohnt sich bei einem langen Wochenende.",
        ],
    },
}

SYSTEM_PROMPT = """Du generierst deutsche Sätze so wie echte Menschen sie auf dem Smartphone tippen.

Wichtige Regeln:
- Natürlich und alltagsnah — kein Lexikon-Stil, kein Wikipedia, kein Nachrichtenstil
- Längenmix: etwa die Hälfte 6–10 Wörter, der Rest 10–16 Wörter
- Korrekte Rechtschreibung und Grammatik
- Keine Anführungszeichen, keine Nummerierung, kein Kommentar
- Genau eine Zeile pro Satz, keine Leerzeilen
- Abwechslungsreich: verschiedene Personen, Kontexte, Tonlagen
- Jeder Satz beginnt mit einem Großbuchstaben"""


def build_prompt(topic: dict, n: int) -> str:
    beispiele = "\n".join(topic["beispiele"])
    return (
        f"Thema: {topic['name']}\n"
        f"Kontext: {topic['beschreibung']}\n\n"
        f"Beispiele für den richtigen Stil:\n{beispiele}\n\n"
        f"Schreib jetzt genau {n} weitere solche Sätze zu diesem Thema. "
        f"Variiere Inhalt, Länge und Perspektive. Einen Satz pro Zeile."
    )


def ollama_chat(host: str, model: str, system: str, user: str, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model,
        "think": False,
        "stream": False,
        "options": {"num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"]


def parse_sentences(raw: str) -> list[str]:
    # Wenn keine Zeilenumbrüche: auf Satzgrenzen aufsplitten
    if '\n' not in raw and len(raw) > 200:
        raw = re.sub(r'([.!?])([A-ZÄÖÜ])', r'\1\n\2', raw)

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-•]\s*', '', line)
        if not line or len(line) < 8 or line.endswith(':'):
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        if line:
            line = line[0].upper() + line[1:]
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-topic", type=int, default=50,
                        help="Sätze pro Thema (default: 50)")
    parser.add_argument("--batch", type=int, default=25,
                        help="Sätze pro API-Aufruf (default: 25)")
    parser.add_argument("--model", default="qwen3.6:27b",
                        help="Ollama-Modellname (default: qwen3.6:27b)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--topics", default=None,
                        help="Kommagetrennte Themen-Keys (z.B. mittelalter,medizin)")
    parser.add_argument("--list-topics", action="store_true",
                        help="Verfügbare Themen auflisten und beenden")
    parser.add_argument("--output-dir", default=str(DATA_DIR),
                        help=f"Ausgabeverzeichnis (default: {DATA_DIR}); "
                             "je Thema eine Datei: synthetic_<key>.txt")
    args = parser.parse_args()

    if args.list_topics:
        print("Verfügbare Themen:")
        for key, t in TOPICS.items():
            print(f"  {key:<20} — {t['name']}")
        return

    out_dir = Path(args.output_dir)

    # Themen auswählen
    if args.topics:
        keys = [k.strip().lower() for k in args.topics.split(",")]
        unknown = [k for k in keys if k not in TOPICS]
        if unknown:
            print(f"Unbekannte Themen: {unknown}", file=sys.stderr)
            print(f"Verfügbar: {list(TOPICS.keys())}", file=sys.stderr)
            sys.exit(1)
        selected = {k: TOPICS[k] for k in keys}
    else:
        selected = TOPICS

    total_target = len(selected) * args.per_topic
    print(f"Themen: {len(selected)}  ×  {args.per_topic} Sätze = ~{total_target} Sätze")
    print(f"Modell: {args.model}  →  {out_dir}/synthetic_<thema>.txt\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_collected = 0
    t_start = time.time()

    # Zähle bereits vorhandene Sätze pro Topic
    counts = {}
    for key in selected:
        out_path = out_dir / f"synthetic_{key}.txt"
        counts[key] = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0

    # Round-Robin: eine Batch pro Topic pro Runde bis alle per_topic erreicht
    topic_keys = list(selected.keys())
    errors_per_topic = {k: 0 for k in topic_keys}

    while any(counts[k] < args.per_topic for k in topic_keys):
        for key in topic_keys:
            if counts[key] >= args.per_topic:
                continue
            topic = selected[key]
            out_path = out_dir / f"synthetic_{key}.txt"
            n = min(args.batch, args.per_topic - counts[key])
            pct = int(counts[key] / args.per_topic * 100)
            print(f"  [{key}] {counts[key]}/{args.per_topic} ({pct}%)", end=" ", flush=True)
            try:
                raw = ollama_chat(
                    host=args.host,
                    model=args.model,
                    system=SYSTEM_PROMPT,
                    user=build_prompt(topic, n),
                    max_tokens=n * 40,
                )
                sentences = parse_sentences(raw)
                with out_path.open("a", encoding="utf-8") as f:
                    for s in sentences:
                        f.write(s + "\n")
                counts[key] += len(sentences)
                total_collected += len(sentences)
                errors_per_topic[key] = 0
                print(f"+{len(sentences)}")
            except KeyboardInterrupt:
                print(f"\nAbgebrochen. {total_collected} Sätze gespeichert.")
                return
            except Exception as e:
                print(f"  Fehler: {e}", file=sys.stderr)
                errors_per_topic[key] += 1
                if errors_per_topic[key] > 3:
                    print(f"  Zu viele Fehler bei '{key}', überspringe.")
                    break
                time.sleep(2)

    elapsed = time.time() - t_start
    print(f"\nFertig: {total_collected} Sätze in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
