# Staged CLI bounded-record admission

No canonical files were changed and no checks, tests, builds, provider calls, or queue jobs were run. Apply the relative full files only if canonical baseline hashes match manifest.json. The new test file must not exist before integration.

The CLI JSON reader now opens with O_NONBLOCK and O_NOFOLLOW when supported, rejects symlinks, and checks the actual opened descriptor for a regular file within the byte limit before reading. It retains the bounded read, JSON constant rejection, object requirement, exact original-byte SHA-256 and error codes. A finite parse_float additionally rejects exponent overflow such as nested positive or negative 1e9999 while preserving finite decimal/exponent floats. Oversized or nonregular records raise invalid_record; OS access errors retain io_error. Existing run-level symlink admission is unchanged.

Tests cover FIFO/directory/symlink config inputs, targeted status and history with nonregular plans, all three optional receipts in both status modes, valid finite floats and exact-byte digests, malformed/nonfinite objects including nested positive/negative exponent overflow, exact/oversized byte bounds, growth after descriptor admission, replacement-by-symlink, and missing-file errors. Nonregular tests guard Path.open, builtin/io open, and os.open flags before attempting the actual special-file access, so a blocking-open regression fails rather than hanging. Explicit downstream guards reject dataset planning, readiness/inspection, launch, credentials, and Popen calls.

This fixes local CLI file admission only. It does not attest remote packages or providers, change workspace readers, or claim training quality or performance improvements.
