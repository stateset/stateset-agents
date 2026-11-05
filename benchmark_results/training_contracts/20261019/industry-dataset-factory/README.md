# Industry dataset factory

Created ecommerce/retail, logistics/fulfillment and SaaS support corpora. Each contains 512 train, 64 validation and 128 reserved test conversations: 2,112 total. All are labeled synthetic and not independently reviewed. Messages teach fact lookup, a policy-grounded JSON resolution and explicit finish; no real accounts, shipments or refunds are changed.

All 2,112 examples passed reconstruction. Every split is balanced across eight scenario families. Deterministic top-up seeds replace repeated fact sets without moving held-out rows into training; manifests bind all source cases and artifacts. Unverified SaaS/logistics lookups reveal only ID and verification status. Policy priority, inclusive thresholds, malicious notes, invalid actions, split leakage, publication failures and modified artifacts are covered.

152 industry tests passed, followed by 26 changed-policy rechecks. Full-package mypy passed for 443 files; Ruff, Black, isort and repository hygiene passed. All three existing industry training previews succeeded with no model loading or provider calls. The industry evaluator continues to reject synthetic data for its quality gate.

Reproduce with `stateset-agents industry generate all --output outputs/industry-datasets --seed 20261019`, then `stateset-agents industry audit-datasets outputs/industry-datasets`. Existing trainer: `stateset-agents industry train outputs/industry-datasets/ecommerce --dry-run` (likewise logistics/saas). Remote SFT takes only train.jsonl. Training is not launched by generation.

The datasets archive contains complete output files. The source archive and source identity bind the reviewed implementation. This review predates Studio import support; the strict text-only picker still rejected provenance-rich rows at this snapshot. These synthetic data and CPU checks do not establish learning, GPU/provider parity, independent production quality or an A+ result.
