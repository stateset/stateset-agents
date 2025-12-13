# StateSet Agents v0.7.1

A training-infrastructure release that makes it easier to build “real RL” workflows:
consistent callbacks, safer concurrent evaluation, and a shared evaluation harness.

## Highlights
- **Unified Training Callback Dispatch**
  - New `stateset_agents.training.callbacks` dispatch layer supports both callable callbacks
    (`callback(event, data)`) and method callbacks (`on_episode_end`, etc.).
  - GRPO trainers now emit consistent lifecycle events: training start/end, episode end,
    evaluation end, and checkpoint saved.
- **Safe Concurrent Rollouts (via Env Cloning)**
  - New `Environment.clone()` contract (implemented for `ConversationEnvironment` and
    `TaskEnvironment`) enables parallel evaluation/rollouts without shared mutable env state.
- **Reusable Evaluation Harness**
  - New `stateset_agents.training.evaluation.evaluate_agent()` + `EvaluationConfig` for
    consistent evaluation metrics across agents/environments.
  - Multi-turn GRPO evaluation now uses this harness and supports configurable concurrency
    and multiple generations per scenario.
- **Config Knobs for Rollouts/Eval**
  - `TrainingConfig.rollout_concurrency` and `TrainingConfig.eval_num_generations` added for
    controlling evaluation behavior.

## Upgrade Notes
```bash
pip install -U stateset-agents==0.7.1
```

If you evaluate with concurrency (`rollout_concurrency > 1`) on a custom environment,
implement `Environment.clone()` or keep concurrency at 1.

## Verification
- `pytest -q`
- `stateset-agents version`
- `stateset-agents train --stub`

## Artifacts
- `dist/stateset_agents-0.7.1-py3-none-any.whl`
- `dist/stateset_agents-0.7.1.tar.gz`

---

## Previous Release: v0.6.0

See `CHANGELOG.md` and the GitHub tag `v0.6.0`.
