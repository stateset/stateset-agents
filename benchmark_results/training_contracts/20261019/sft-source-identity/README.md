# Captured source identity for managed SFT

River SFT now hashes the same captured bytes it parses before client checks and
tokenization. Fetch uses that retained original path, SHA-256 and nonblank
source-line count, even if the file is replaced, deleted or the caller repoints
the mutable spec. It no longer hashes a later version as training evidence.

Fireworks validates captured bytes and uploads those bytes through a private
temporary file, which is removed after success or an upload exception. The
source identity persists with its durable job metadata. Restarted fetch validates
the retained configuration without requiring the original dataset to exist;
normal new-job construction retains its existing file-existence requirement.
Legacy metadata lacking a captured identity stays fetchable with an explicitly
unknown hash/count rather than adopting the current file's contents.

489 unique tests passed: 74 source/split boundary checks, including 22 new source
identity tests, and 415 provider, job-spec, SFT, lineage and CLI regressions. Tests
replace/delete the source or repoint the spec during provider callbacks, capture
uploaded bytes, verify private-file cleanup on transport failure, reject split
changes between preflight and capture, validate retained identity records before
provider polling, and exercise restart and legacy metadata behavior. Full-package
mypy passed for 444 files. Ruff, Black, isort, repository hygiene and whitespace
checks passed; Black ran outside the known sandbox worker-IPC restriction.

The corpus review parsed all 1,536 retained industry training rows through both
captured-byte readers, matched their existing dataset manifest hashes and
rejected 12 held-out corpus/reader combinations. It made no provider, tokenizer
or training calls. Source identities and selected sources/tests are archived;
all previous retained evidence artifacts were checked without modifying them.

This is local source lineage, not provider attestation or proof of learning.
Source-line counts can exceed effective training targets after parsing,
rendering, filtering or truncation. The provider regression suites use recording
fakes; this review adds no live SDK/service certification or paid run. Native RL,
other provider upload paths and generic local SFT manifests retain their separate
protocols. Changes before byte capture are not bound to an earlier review hash,
and independently edited local metadata is not authenticated. These checks do
not establish reviewed production data, model-quality gains or A+ status.
