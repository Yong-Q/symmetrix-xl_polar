#!/usr/bin/env bash
# veMLP-style entry point: run one shard of the density scan on a single GPU.
#
#   MANIFEST=.../manifest.remote.csv  WORKDIR=... bash jobs/entrypoint_scan.sh
#
# Environment (defaults in brackets):
#   BUNDLE        bundle root                     [dir containing this script/..]
#   WORKDIR       task working directory          [$PWD]
#   MANIFEST      shard manifest (csv)            [$WORKDIR/manifest.remote.csv]
#   OUTROOT       where samples/ go               [$WORKDIR/samples]
#   EQUIL_STEPS   equilibration steps             [200000]
#   STEPS         production steps                [2000000]
set -euo pipefail
_here=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
BUNDLE=${BUNDLE:-$(cd -- "$_here/.." && pwd -P)}
. "$BUNDLE/env.sh"

WORKDIR=${WORKDIR:-$PWD}
MANIFEST=${MANIFEST:-$WORKDIR/manifest.remote.csv}
OUTROOT=${OUTROOT:-$WORKDIR/samples}
ECHO_STEPS=${EQUIL_STEPS:-200000}
PROD_STEPS=${STEPS:-2000000}

cd "$WORKDIR"
echo "== bundle    : $MACELES_HOME"
echo "== manifest  : $MANIFEST"
echo "== outroot   : $OUTROOT"
echo "== lmp       : $LMP"
echo "== cuda libs : $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -m1 cuda || echo '(from ldconfig)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 "$BUNDLE/sampling/les_batch.py" \
    --manifest "$MANIFEST" \
    --outroot "$OUTROOT" \
    --run --resume --no-analyze \
    --equil-steps "$ECHO_STEPS" --steps "$PROD_STEPS" \
    --lmp "$LMP" --model "$MODEL" --cubin "$CUBIN" \
    --ld-library-path "$LD_LIBRARY_PATH"

echo "== done; samples: $(find "$OUTROOT" -name meta.json | wc -l) states =="
