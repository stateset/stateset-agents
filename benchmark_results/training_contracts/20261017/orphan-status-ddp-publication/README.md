# Orphan outcomes, bounded status and collective checkpoint publication

Buildr `0de5107bf1dc` passed 260 native checkpoint tests,
824 operator tests, 3 real two-worker
CPU/Gloo contracts and 8 governance tests, plus typing
for 431 package modules, formatting, Ruff, repository hygiene, stable
API/release governance and workflow lint. No selected cases failed or skipped.
Another 93 overlapping cases passed from an installed wheel with
identical source bytes, checked import origins and matching JUnit case identities.
This is targeted verification; it is not a full repository test run.

Workers commit success after checked completion publication, including after
supervisor loss, without inventing an observed exit code. Serialized bounded
state transitions prevent later progress or completion from replacing an already
recorded failure/cancellation, including both concurrent ordering cases. Unfinished direct
ownership loss stays unconfirmed, and completion files alone imply no success.
Config and status readers reject nonregular files without blocking and retain
exact-byte checksums. Tests use actual control paths with injected training and
provider fixtures; no provider was contacted.

Both native collective checkpoint APIs now coordinate strict rollout publication
and restore each rank's RNG after publication before clearing quarantine. Two
live CPU/Gloo workers cover acknowledged, legacy None, absent and disabled sync,
asymmetric False/errors/final RNG faults, refusal while quarantined, healthy
reload, distinct Python/NumPy/Torch streams and exact stochastic next updates.
The existing broad native checkpoint two-worker regression also passed.
This contains failed application; it does not roll back partly applied state or
recover lost peers. Legacy None has no positive publication acknowledgement.

Source/test hashes were frozen after formatting and unchanged throughout checks.
The refreshed live pane matches them; saved draft bytes, permissions and run
history were preserved and no training was launched. Staged manifests describe
preformatter inputs; verified bytes are retained separately.

Broader source/wheel/CLI acceptance is separate under `1e204ba66630` (state at
retention: succeeded); consult its source hashes for scope. This receipt
adopts no GPU, provider-training or model-quality evidence.
