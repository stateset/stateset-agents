River AI 0.11 Provider
======================

StateSet Agents integrates end-to-end with **River AI (``river-client`` 0.11)**
for managed open-weight SFT, rejection-sampling flywheels, group-relative RL,
native ``rl.RolloutEngine`` sandboxes, and hosted checkpoint serving/deployments.

* **Full Provider Reference (Markdown):** `docs/RIVER_PROVIDER.md <https://github.com/stateset/stateset-agents/blob/master/docs/RIVER_PROVIDER.md>`_
* **Getting Started (Markdown):** `docs/GETTING_STARTED_RIVER.md <https://github.com/stateset/stateset-agents/blob/master/docs/GETTING_STARTED_RIVER.md>`_

Install & Preflight
-------------------

River's SDK requires Python ``>=3.12`` (the rest of ``stateset-agents`` supports
Python ``>=3.10``):

.. code-block:: bash

   pip install "stateset-agents[river]"
   export RIVER_API_KEY=rv_...
   stateset-agents river-preflight --live

Training, RL & Hosted Serving CLI
---------------------------------

.. code-block:: bash

   # Managed SFT & RL (cispo, ppo, decoupled_ppo, dro, importance_sampling)
   stateset-agents train-remote --provider river --dataset train.jsonl \
     --base-model Qwen/Qwen3.6-35B-A3B-FP8 --output-dir outputs/my_model

   # Query a hosted river:// checkpoint pointer directly
   stateset-agents river-chat --checkpoint outputs/my_model --prompt "Refund order ORD-77701?"

   # Create, meter, scale, promote streaming replicas, and delete deployments
   stateset-agents river-deploy --checkpoint outputs/my_model --unified-replicas 1 --wait
   stateset-agents river-deploy --checkpoint outputs/my_model --promote-streaming-model my-agent-live
   stateset-agents river-deployments --deployment-id dep_123 --usage
   stateset-agents river-deployments --deployment-id dep_123 --scale-unified 2
   stateset-agents river-deployments --deployment-id dep_123 --delete

Programmatically, ``RiverDeploymentManager``, ``RiverDeploymentProvider``
(``deploy-inference --provider river``), and ``RiverInferenceClient`` live in
``stateset_agents.remote.river_serving``, while ``river_mcp_sandbox_factory`` in
``stateset_agents.remote.river_environment`` bridges all six enterprise
verifiable sandboxes into River's native ``rl.RolloutEngine``.
