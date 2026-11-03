# Source this file to use a portable MACELES/LAMMPS engine bundle.
#
#   . /path/to/maceles-lmp-env/env.sh
#   lmp-run -in examples/smoke/in.smoke -var home "$MACELES_HOME"
#
# Everything the binary needs beyond glibc/the NVIDIA driver is inside lib/;
# libcublas is the one exception (105 MB), so we look for a CUDA toolkit that
# has it and prepend that directory.
_mlh_src=${BASH_SOURCE[0]:-$0}
MACELES_HOME=$(cd -- "$(dirname -- "$_mlh_src")" && pwd -P)
export MACELES_HOME
export PATH="$MACELES_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MACELES_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$MACELES_HOME${PYTHONPATH:+:$PYTHONPATH}"
export LMP="$MACELES_HOME/bin/lmp"
export MODEL="$MACELES_HOME/assets/maceles_off/maceles_off_chno.json"
export CUBIN="$MACELES_HOME/assets/maceles_off/maceles_off.cubin"

_mlh_find_cublas() {
    # $1 directory, ok if it holds libcublas.so.12
    [ -n "${1:-}" ] && [ -e "$1/libcublas.so.12" ]
}
_mlh_cuda=""
for _d in ${MACELES_CUDA_LIB:-} ${CUDA_HOME:+$CUDA_HOME/lib64} \
          /opt/matos-tools/cuda/*/lib64 /usr/local/cuda*/lib64 \
          /usr/local/cuda/lib64; do
    if _mlh_find_cublas "$_d"; then _mlh_cuda=$_d; break; fi
done
if [ -n "$_mlh_cuda" ]; then
    case ":$LD_LIBRARY_PATH:" in
        *":$_mlh_cuda:"*) ;;
        *) export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_mlh_cuda" ;;
    esac
    export CUDA_HOME=${_mlh_cuda%/lib64}
elif ! ldconfig -p 2>/dev/null | grep -q 'libcublas\.so\.12'; then
    echo "warning: libcublas.so.12 not found (set CUDA_HOME or MACELES_CUDA_LIB)" >&2
fi
unset -f _mlh_find_cublas
