#!/usr/bin/env sh
set -eu

RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }--cfg docsrs -D warnings" \
    exec cargo +nightly doc --all-features --no-deps "$@"
