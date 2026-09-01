# API compatibility policy

StateSet Agents freezes a machine-readable v1 contract for its supported
Python and HTTP surfaces. The guarantee begins with the release that includes
[`contracts/public_api_v1.json`](../contracts/public_api_v1.json); the `v1`
label describes the public contract and does not imply that the Python package
has reached version 1.0.

## Covered surface

The contract records:

- every public symbol exported by `stateset_agents.__all__`, including its
  import target and inspectable parameter names, calling kinds, and defaults;
- every public symbol exported by `stateset_agents.api.__all__`, with the same
  signature protection;
- the production `/v1/*`, `/api/v1/*`, health, readiness, and liveness
  operations listed in `scripts/check_api_compatibility.py`; and
- the request, response, parameter, and referenced component schemas that
  determine those operations' wire behavior.

Descriptions, examples, summaries, and generated schema titles are excluded so
documentation can improve without creating false compatibility failures.
`stateset_agents.experimental`, deprecated shim packages, and the opt-in
`/api/lab/*` training-lab router are not stable surfaces.

## Guarantee

Existing stable imports and HTTP wire contracts remain compatible within the
v1 line. A breaking change must use a new versioned surface. When a stable
Python symbol can be retired without a security or correctness emergency, it
is first deprecated in a documented release and remains available for at
least one subsequent minor release.

Additive changes are allowed only after deliberate review. They update the
manifest so an accidental export or route cannot silently become a permanent
support obligation. A reviewed contract change must include tests,
documentation, and a changelog entry. A breaking migration must also include
an old-to-new mapping and an overlap period where both surfaces work whenever
that is technically safe.

## Maintainer workflow

Run the gate locally with:

```bash
make api-compatibility
```

After intentionally changing a stable surface, regenerate the contract and
review the complete diff:

```bash
python scripts/check_api_compatibility.py --write
git diff -- contracts/public_api_v1.json
make api-compatibility
```

The checker performs an exact structural comparison and exits nonzero on any
unreviewed addition, removal, or change. CI and the publish-readiness gate run
the same check, so a release cannot bypass the committed contract.
