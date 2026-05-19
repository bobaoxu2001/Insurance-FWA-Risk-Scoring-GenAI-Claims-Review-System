#!/usr/bin/env bash
# Download the two real public datasets used by this project:
#   1. HHS-OIG List of Excluded Individuals/Entities (LEIE)  — 15 MB
#   2. CMS Nursing Home Provider Information                 —  9 MB
#
# Both are public, non-licensed, updated regularly.
# This script is idempotent — skips files that are already on disk.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw/oig data/raw/cms

LEIE_FILE="data/raw/oig/oig_leie_updated.csv"
LEIE_URL="https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"

CMS_FILE="data/raw/cms/cms_nursing_home_provider_info.csv"

echo "=== 1. HHS-OIG LEIE ==="
if [ -s "$LEIE_FILE" ]; then
  echo "  Already present: $LEIE_FILE ($(du -h "$LEIE_FILE" | cut -f1))"
else
  echo "  Downloading from $LEIE_URL ..."
  curl -sL -o "$LEIE_FILE" "$LEIE_URL"
  echo "  Done: $(du -h "$LEIE_FILE" | cut -f1)"
fi

echo ""
echo "=== 2. CMS Nursing Home Provider Information ==="
if [ -s "$CMS_FILE" ]; then
  echo "  Already present: $CMS_FILE ($(du -h "$CMS_FILE" | cut -f1))"
else
  echo "  Paginated download via CMS provider-data API (~15 calls, ~30s) ..."
  python3 - <<'PY'
import requests, time
URL = "https://data.cms.gov/provider-data/api/1/datastore/query/4pq5-n9py/0"
out = "data/raw/cms/cms_nursing_home_provider_info.csv"
parts, offset = [], 0
while True:
    r = requests.get(URL, params={"format": "csv", "limit": 1000, "offset": offset}, timeout=60)
    r.raise_for_status()
    text = r.text
    if not text.strip():
        break
    if offset == 0:
        parts.append(text)
    else:
        parts.append("\n".join(text.splitlines()[1:]))
    n = len(text.splitlines()) - 1
    print(f"    offset={offset}  rows={n}")
    if n < 1000:
        break
    offset += 1000
    time.sleep(0.2)
with open(out, "w") as f:
    f.write("\n".join(parts))
PY
  echo "  Done: $(du -h "$CMS_FILE" | cut -f1)"
fi

echo ""
echo "Both real datasets ready. Next:"
echo "  python src/oig_leie_analysis.py"
echo "  python src/cms_ltc_pipeline.py"
