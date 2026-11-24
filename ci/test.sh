#!/usr/bin/env sh
set -eu

: "${BLAS_LIB:=openblas}"
: "${BLAS_LIB_DIR:=}"

if [ -n "$BLAS_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${BLAS_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export RUSTFLAGS="-L ${BLAS_LIB_DIR} -l ${BLAS_LIB}${RUSTFLAGS:+ $RUSTFLAGS}"
else
    export RUSTFLAGS="-l ${BLAS_LIB}${RUSTFLAGS:+ $RUSTFLAGS}"
fi

cargo test --tests --all-features
cargo +nightly miri test --no-default-features --features alloc
