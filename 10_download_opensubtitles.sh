#!/usr/bin/env bash
# Downloads the German OpenSubtitles 2018 monolingual corpus (~475 MB gz).
set -euo pipefail

URL="https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/de.txt.gz"
OUT="data/opensubtitles/de.txt.gz"

mkdir -p data/opensubtitles

if [[ -f "$OUT" ]]; then
    echo "Bereits heruntergeladen: $OUT"
    exit 0
fi

echo "Lade OpenSubtitles DE herunter (~475 MB) ..."
wget -c --progress=bar:force "$URL" -O "$OUT"
echo "Fertig: $OUT"
