#!/usr/bin/env sh
set -eu

for std in "" "alloc" "std"; do
    for gemm in "" "backend-gemm" "gemm-f16"; do
        for mm in "" "backend-matrixmultiply"; do
            for blas in "" "backend-blas"; do
                feats=$(printf "%s %s %s %s" "$std" "$gemm" "$mm" "$blas" | xargs)
                echo "==> clippy --no-default-features --features \"$feats\""
                cargo clippy --all-targets --no-default-features --features "$feats" -- -D warnings
                cargo doc --no-deps --no-default-features --features "$feats"
            done
        done
    done
done

cargo clippy -p tensor-benches --all-targets -- -D warnings
