fn main() {
    println!("cargo:rerun-if-env-changed=BLAS_LIB");
    println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");

    if let Ok(dir) = std::env::var("BLAS_LIB_DIR")
        && !dir.is_empty()
    {
        println!("cargo:rustc-link-search=native={dir}");
    }
    let lib = std::env::var("BLAS_LIB").unwrap_or_else(|_| "openblas".into());
    println!("cargo:rustc-link-lib={lib}");
}
