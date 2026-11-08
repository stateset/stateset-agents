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

Subpackage namespaces such as `stateset_agents.training` (whose `__all__`
re-exports several hundred trainers, configs, and helpers) are **provisional**:
they are importable and tested, but only names re-exported from
`stateset_agents.__all__` carry the compatibility guarantee. Import stable
entry points from the top-level package; reach into subpackages when you need
something the stable surface does not expose and pin the release.

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

## Runtime dependency migrations after v0.72.0

The weight-editing follow-up removes six compatibility bridges that had reached
their v0.72.0 deadline. The supported root exports and HTTP contract are
unchanged. Callers using the retired implicit behavior or import aliases must
use the following explicit dependencies:

| Retired path or behavior | Replacement |
| --- | --- |
| `MultiTurnAgent(config)` with planning enabled and no manager | Pass `planning_manager=planning_manager_from_config(config.planning_config)` from `experimental.long_term_planning` |
| Implicit planning in the deprecated `DialogueSessionAgent` | Inject `planning_manager=` explicitly; prefer the supported `core.agent.MultiTurnAgent` for new integrations |
| `train()` implicitly loading an offline trainer | Import `stateset_agents.experimental.offline_rl` first, or register a custom loader with `training.offline_backend.register_offline_trainer` |
| `core.types.TrainingConfig` | `stateset_agents.TrainingConfig` or `stateset_agents.training.config.TrainingConfig` |
| `core.enhanced.advanced_rl_algorithms.GSPOConfig` | `stateset_agents.training.gspo_config.GSPOConfig` |
| `evaluation.reward_audit.audit_support_rewards` and its old CLI | `stateset_agents.rewards.support_reward_audit.audit_support_rewards` and `python -m stateset_agents.rewards.support_reward_audit` |

For planning:

```python
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.experimental.long_term_planning import planning_manager_from_config

config = AgentConfig(
    model_name="stub://planning", use_stub_model=True,
    enable_planning=True, planning_config={"max_steps": 3},
)
agent = MultiTurnAgent(
    config, planning_manager=planning_manager_from_config(config.planning_config),
)
```

`enable_planning=True` without a manager now raises `ConfigurationError` and
names the factory; invalid planning requests cannot silently disable planning.
The gateway already injects the planner, so its request schema and supported
planning behavior are preserved.

For offline training, import the experimental component before the existing
training call:

```python
import stateset_agents.experimental.offline_rl  # Registers its lazy trainer loader.
from stateset_agents import train

# In an async function, with an initialized agent, environment, and logged data:
# await train(agent, environment, training_mode="offline", dataset=logged_data)
```

Hybrid training requires the same registration when it has an offline dataset.
Hybrid mode without a dataset can still run its online phase alone. The generic
`evaluation.reward_audit.audit_reward` API remains available; invoking its retired
support-reward CLI exits with a message naming the replacement command.

This completes the six runtime dependency migrations, not the entire v0.72.0
deprecation schedule. Other overdue aliases and modules remain, including root
exports, compatibility shims, and deprecated application surfaces. The global
deprecation-date check continues to report them; no removal deadlines have been
extended or exempted to make that check pass.
