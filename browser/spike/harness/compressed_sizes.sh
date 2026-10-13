#!/usr/bin/env bash
# Raw, gzip -9 and brotli sizes of the files a browser would download (docs/m0/browser-wasm-spike.md).
# Brotli quality 11 for text assets, 9 for graphs (11 on hundreds of MB takes too long to be a realistic build step).
# Usage: compressed_sizes.sh <quality> <file>...
set -euo pipefail
quality="$1"
shift
for f in "$@"; do
  printf '%s raw=%s gzip9=%s br%s=%s\n' "$(basename "$f")" "$(stat -c %s "$f")" "$(gzip -9c "$f" | wc -c)" \
    "$quality" "$(brotli -c -q "$quality" "$f" | wc -c)"
done
