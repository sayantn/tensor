#!/usr/bin/env sh
set -eu

export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -n "${BLAS_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${BLAS_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export RUSTFLAGS="-C target-cpu=native ${RUSTFLAGS:-}"

cargo bench -p tensor-benches "$@"
