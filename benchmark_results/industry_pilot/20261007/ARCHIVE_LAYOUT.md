# Historical environment snapshot filenames

The `attempt1`, `attempt2`, and `attempt5` environment snapshots use the
canonical filename `pip-freeze-observed.txt`. Upstream renamed the original
`requirements-observed.txt` files so GitHub's dependency graph would not treat
the pod's historical environment as the project's current dependency manifest.

The original and canonical files have identical bytes and SHA-256
`8a0efd352e8b2f95e2273a78d992ea75ff9ac659143406df892140d80ebe4c2c`.
`checksums.json` uses the canonical filenames. Retained launcher source still
shows the name written during the original run.

The original `README.md` is itself bound by `checksums.json`. Its historical
bytes remain unchanged; this separate note records the layout migration without
changing the report, measurements, or the retained environment snapshots.
