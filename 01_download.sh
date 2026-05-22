#!/usr/bin/env bash
# Downloads the latest German Wikipedia articles dump.
set -euo pipefail

DUMP_URL="https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2"
OUT="data/dewiki-latest-pages-articles.xml.bz2"

mkdir -p data

if [[ -f "$OUT" ]]; then
    echo "Dump already exists: $OUT"
    exit 0
fi

echo "Downloading German Wikipedia dump (~6 GB)..."
wget -c --progress=bar:force "$DUMP_URL" -O "$OUT"
echo "Done: $OUT"
