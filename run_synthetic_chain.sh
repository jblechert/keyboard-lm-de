#!/usr/bin/env bash
# Wartet bis synthetic_de.txt 20k Sätze hat, dann startet topic-spezifische Generation.
set -euo pipefail

TARGET=20000
GEN_PID=411417
PER_TOPIC=1000

echo "=== Synthetic Chain ==="
echo "Warte auf ${TARGET} Sätze in data/synthetic_de.txt ..."
echo "(Aktuell: $(wc -l < data/synthetic_de.txt))"

while true; do
    current=$(wc -l < data/synthetic_de.txt)
    if [ "$current" -ge "$TARGET" ]; then
        echo ""
        echo "=== ${current} Sätze erreicht — stoppe allgemeinen Generator ==="
        kill "$GEN_PID" 2>/dev/null && echo "PID ${GEN_PID} gestoppt." || echo "PID ${GEN_PID} bereits beendet."
        break
    fi
    echo "  $(date +%H:%M)  ${current}/${TARGET} ..."
    sleep 120
done

echo ""
echo "=== Starte topic-spezifische Generation (${PER_TOPIC} Sätze/Thema, alle Topics) ==="
.venv_ml/bin/python 12_generate_synthetic_vocab.py \
    --per-topic "$PER_TOPIC" \
    --batch 25

echo ""
echo "=== Fertig. Topic-Dateien in data/: ==="
ls -lh data/synthetic_*.txt
