# Rust crates in this repository

This repository contains two independent Rust crates that are easy to conflate. They are unrelated in
purpose, dependencies, and publication status.

## `rust_core` — `stateset-rl-core`

Path: `rust_core/`

`rust_core` is the pyo3 accelerator for the Python `stateset-agents` package. It provides high-performance
implementations of RL-training hot paths (GAE, group-relative advantages, GSPO importance ratios, reward
normalization) and is built as a Python extension module with [maturin](https://www.maturin.rs/).

- Published to PyPI as `stateset-rl-core` and to crates.io as `stateset-rl-core`.
- The pyo3/numpy bindings live behind the optional `python` Cargo feature, so `cargo check`/`cargo test`
  work without libpython, and `docs.rs` can build the crate. `maturin build` enables the `python` feature
  (see `rust_core/pyproject.toml`'s `[tool.maturin] features`).
- Installed by end users via `pip install stateset-agents[rust]` (or `[full]`).

## Root crate — the StateSet commerce daemon

Path: `Cargo.toml`, `src/` (repository root)

The root crate (package name `stateset-agents`, binary `stateset-agents`) is an internal daemon that
integrates with the StateSet commerce API over REST and gRPC. It is **not** the Python package of the same
name, and despite sharing a name it is unrelated to PyPI's `stateset-agents`.

- `publish = false` — this crate is never published to crates.io. It needs a sibling `stateset-api` repo
  to build/run meaningfully, and the crates.io name `stateset-agents` would collide with the unrelated PyPI
  package.
- It is for internal StateSet use only; there is no public release process for it.

## Summary

| Crate | Path | Package name | Published? | Purpose |
|---|---|---|---|---|
| `rust_core` | `rust_core/` | `stateset-rl-core` | Yes (PyPI + crates.io) | pyo3 accelerator for the Python framework |
| root crate | `/` | `stateset-agents` | No (`publish = false`) | Internal StateSet commerce daemon |
