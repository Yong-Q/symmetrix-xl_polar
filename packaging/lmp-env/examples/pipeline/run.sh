#!/usr/bin/env bash
# Full chain smoke test: build box -> MD via les_batch -> structure/c1c2 post-processing.
# Needs Python 3.11 with numpy scipy ase matplotlib; takes ~2 minutes on one A100.
#
#   OUT=/tmp/smoke examples/pipeline/run.sh
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUNDLE=$(cd -- "$HERE/../.." && pwd -P)
. "$BUNDLE/env.sh"
PY=${PY:-python3}
OUT=${OUT:-$PWD/smoke}
mkdir -p "$OUT" && cd "$OUT"

echo "== 1. build a 64-molecule water box =="
"$PY" -m sampling.make_smoke_box --out "$OUT"

echo "== 2. MD (500 steps, dump every 100) =="
"$PY" "$BUNDLE/sampling/les_batch.py" \
    --manifest "$OUT/manifest.csv" --outroot "$OUT/samples" \
    --run --resume --no-analyze \
    --equil-steps 200 --steps 500 \
    --lmp "$LMP" --model "$MODEL" --cubin "$CUBIN" \
    --ld-library-path "$LD_LIBRARY_PATH"

echo "== 3. post-process =="
"$PY" "$BUNDLE/sampling/scan_analysis.py" --root "$OUT" --out "$OUT/analysis_scan" \
    --frames 10 --c2-stride 1 --c2-bins 40

echo "== 4. result =="
cat "$OUT/analysis_scan/scan_summary.csv"
ls "$OUT/analysis_scan/figures"
echo "pipeline OK"
