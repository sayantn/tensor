#!/usr/bin/env sh
set -eu

RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }--cfg docsrs -D warnings" \
    cargo +nightly doc --all-features --no-deps "$@"
