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

OUT = Path("data/synthetic_themen.txt")

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
- Abwechslungsreich: verschiedene Personen, Kontexte, Tonlagen"""


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
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-•]\s*', '', line)
        if not line or len(line) < 8 or line.endswith(':'):
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
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
    parser.add_argument("--output", default=str(OUT),
                        help=f"Ausgabedatei (default: {OUT})")
    args = parser.parse_args()

    if args.list_topics:
        print("Verfügbare Themen:")
        for key, t in TOPICS.items():
            print(f"  {key:<20} — {t['name']}")
        return

    out_path = Path(args.output)

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
    print(f"Modell: {args.model}  →  {out_path}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_collected = 0
    t_start = time.time()

    with out_path.open("a", encoding="utf-8") as f:
        for i, (key, topic) in enumerate(selected.items()):
            collected = 0
            errors = 0
            print(f"[{i+1}/{len(selected)}] {topic['name']} …")

            while collected < args.per_topic:
                n = min(args.batch, args.per_topic - collected)
                try:
                    raw = ollama_chat(
                        host=args.host,
                        model=args.model,
                        system=SYSTEM_PROMPT,
                        user=build_prompt(topic, n),
                        max_tokens=n * 40,
                    )
                    sentences = parse_sentences(raw)

                    for s in sentences:
                        f.write(s + "\n")
                    f.flush()
                    collected += len(sentences)
                    total_collected += len(sentences)
                    print(f"  {collected}/{args.per_topic} Sätze")

                except KeyboardInterrupt:
                    print(f"\nAbgebrochen. {total_collected} Sätze gespeichert.")
                    return
                except Exception as e:
                    print(f"  Fehler: {e}", file=sys.stderr)
                    errors += 1
                    if errors > 3:
                        print(f"  Zu viele Fehler bei '{key}', überspringe.")
                        break
                    time.sleep(2)

    elapsed = time.time() - t_start
    print(f"\nFertig: {total_collected} Sätze in {elapsed/60:.1f} min → {out_path}")


if __name__ == "__main__":
    main()
