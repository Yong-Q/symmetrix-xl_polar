#!/usr/bin/env bash
# Run the bundled smoke test (2 minutes on an A100) in the current directory.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUNDLE=$(cd -- "$HERE/../.." && pwd -P)
. "$BUNDLE/env.sh"
echo "lmp   = $LMP"
echo "model = $MODEL"
echo "cubin = $CUBIN"
exec lmp-run -k on g 1 -in "$HERE/in.smoke" -var home "$MACELES_HOME" -log log.smoke
