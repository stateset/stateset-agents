# Security policy

StateSet Agents accepts responsible vulnerability reports through private
channels. Do not open a public issue for a suspected vulnerability.

## Supported versions

Security fixes are provided for the latest `0.x` minor release. Older minor
release lines are unsupported; users should upgrade before requesting a fix.

## Reporting a vulnerability

Use either of these private channels:

- email `security@stateset.ai`; or
- GitHub private vulnerability reporting from this repository's **Security**
  tab by selecting **Report a vulnerability**.

Include the affected version, impact, reproduction steps or proof of concept,
known mitigations, and a safe way to contact you. Never include production
credentials, customer data, or destructive payloads.

## Response service levels

These are maximum targets measured from receipt of a reproducible private
report. They are operational targets rather than a guarantee of bounty payment.

| Milestone | Target |
|---|---:|
| Acknowledge report | 24 hours |
| Complete initial severity triage | 72 hours |
| Status updates for critical/high issues | Every 72 hours |
| Critical remediation target | 7 calendar days |
| High remediation target | 14 calendar days |
| Medium remediation target | 30 calendar days |
| Low remediation target | 90 calendar days |

If a target cannot be met, the security team will explain why, provide a
mitigation when one is available, and give the reporter a revised date.

## Severity, remediation, and disclosure

Severity is assessed using impact, exploitability, affected configurations,
and exposure of confidentiality, integrity, or availability. We use CVSS as an
input, not as a substitute for deployment context.

Accepted reports are reproduced privately, fixed on supported versions,
regression-tested, and published through a GitHub Security Advisory. We request
coordinated disclosure until a fix or mitigation is available, with a maximum
embargo target of 90 days unless both parties agree that users are safer with a
different schedule. Advisories credit reporters who want attribution and state
affected versions, remediation, and known limitations.

## Safe harbor

StateSet will not pursue legal action against good-faith research that:

- avoids privacy violations, service disruption, persistence, and data loss;
- uses only accounts and data the researcher owns or is authorized to test;
- stops after demonstrating the minimum evidence needed;
- reports findings promptly through a private channel; and
- allows reasonable time for remediation before disclosure.

This safe harbor does not authorize testing third-party systems, social
engineering, denial of service, physical attacks, or access beyond what is
necessary to demonstrate the issue. It is not a bug-bounty promise.

## Important trust boundaries

- Checkpoints load with `weights_only=True` by default. Pass `trusted=True`
  only for artifacts you produced or independently trust; pickle-capable model
  files can execute code.
- Redis-backed caches deserialize trusted internal values. Only connect them to
  a Redis deployment whose writers are fully controlled.
- `API_REQUIRE_AUTH` must remain enabled in production. Disabling it is a
  development-only configuration.
- Provider credentials must enter through secret references or environment
  variables and must never be committed to configurations or evidence files.
- Tools invoked by an agent carry the caller's side-effect authority. Training
  and evaluation should use read-only or sandboxed tools.

## Security controls and updates

CI runs dependency review, secret detection, CodeQL, Bandit, Safety, Trivy, and
Rust advisory checks where applicable. Release readiness fails closed on high
severity dependency findings. Security fixes and disclosures are published as
GitHub Security Advisories.

The machine-readable response commitments live in
[`contracts/security_response_v1.json`](contracts/security_response_v1.json).
The repository currently has no retained independent third-party security
review; that remains an explicit roadmap gate.
