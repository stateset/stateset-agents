Security
========

Secret configuration
--------------------

``SecureConfig.set_secret`` encrypts values by default. Set
``CONFIG_ENCRYPTION_KEY`` and install ``cryptography`` before storing or reading
encrypted values. The operation raises ``RuntimeError`` if either requirement
is missing; it never substitutes base64 encoding for encryption. Use
``encrypt=False`` only when plaintext storage is explicitly intended. Secrets
are held in memory; ``save_config`` writes their names but not their values.

Gateway authentication
----------------------

Gateway auth is controlled via environment variables such as:

- `API_REQUIRE_AUTH`
- `API_KEYS`
- `API_JWT_SECRET` (required in production)

Security event retention
------------------------

``SecurityMonitor`` retains the most recent 10,000 events by default. Pass a
positive ``max_events`` value to choose another in-memory limit. Recent-event
queries and anomaly detection operate on the retained events.

The API authentication failure tracker retains at most 10,000 credential keys
by default. When full, it prunes expired records before admitting a new key;
if all records are still active, a new invalid credential is rejected without
adding state. Existing lockouts remain in force.

Scanner evidence and CI
-----------------------

The security workflow retains one JSON report, a diagnostic log, and the actual
process exit code for each of Bandit, pip-audit, Semgrep, and Trivy. It evaluates
the reports even after an earlier workflow failure and uploads the evidence
even when the gate rejects it. A successful upload does not imply a clean scan.

``python -m scripts.security_workflow_gate --reports-dir PATH`` checks a retained
bundle. Missing or malformed reports, scanner failures, incomplete Semgrep
scans, and unrecognized severities fail the gate. Exit code 1 is accepted only
for scanners that use it for findings, and only when the report contains
findings. Trivy is run without a findings exit-code override and must exit 0.
CI also passes ``--source-root stateset_agents`` to require every nonempty
packaged Python file in Semgrep's scanned-file list. Ignored or unreadable
package sources cannot silently disappear from that coverage check.

Bandit findings of medium severity or higher block release. Semgrep and Trivy
findings of high severity or higher block the workflow. Semgrep's older
``ERROR``, ``WARNING``, and ``INFO`` levels map to ``HIGH``, ``MEDIUM``, and
``LOW`` respectively. Trivy secrets and failed misconfigurations are checked
alongside vulnerabilities. Dependency exceptions remain restricted to the
reviewed package, version, advisory aliases, and expiry in
``scripts/security_exceptions.py``. Findings and counts are summarized in
``security-gate.json`` without copying matched source or secret values.

The Semgrep scan uses the explicit ``p/security-audit`` ruleset with metrics and
version checks disabled. ``--config=auto`` requires metrics and cannot be used
with this configuration. These rules are retrieved from the Semgrep registry;
pin an exported ruleset as well when an identical future rule set is required.

On hosts where the native Semgrep wheel is incompatible, the official container
provides a supported execution path without downgrading the development tool's
security version floor. This tested image pins Semgrep 1.175.0; it mounts source
read-only and does not forward provider environment variables::

    scan_exit_code=0
    docker run --rm --user "$(id -u):$(id -g)" \
      --cap-drop=ALL --security-opt no-new-privileges \
      -e SEMGREP_SETTINGS_FILE=/tmp/semgrep-settings.yml \
      -e SEMGREP_LOG_FILE=/tmp/semgrep.log -e XDG_CACHE_HOME=/tmp/semgrep-cache \
      --mount "type=bind,src=$PWD,dst=/src,readonly" --workdir /src \
      --entrypoint semgrep \
      semgrep/semgrep@sha256:b94b53d02fd4a022f9eac4e2af1380f5c3c4c21400e79d3336bdff1d1db5e796 \
      scan --config=p/security-audit --metrics=off --disable-version-check \
      --jobs=2 --timeout=30 --strict --error --json stateset_agents \
      > /tmp/semgrep-report.json 2> /tmp/semgrep-report.txt || scan_exit_code=$?
    printf '%s\n' "$scan_exit_code" > /tmp/semgrep-exit-code.txt

This command scans packaged Python sources. The CI workflow scans the checkout
subject to its ignore rules. Neither a clean static scan nor passing unit tests
establishes production security or live training effectiveness.
