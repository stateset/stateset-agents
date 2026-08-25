GLM 5.1 Starter Path
====================

Use this starter when you want the fastest path to a first GSPO post-training
run for ``zai-org/GLM-5.1``. GLM 5.1 is a 754B-parameter Mixture-of-Experts
model with DeepSeek V3-style Multi-head Latent Attention (MLA) and 256 routed
experts (8 active per token). A private FP8 deployment alias such as
``your-org/GLM-5.1-FP8`` requires a full 8×H200/B200 host, so this starter
assumes QLoRA training and vLLM-backed serving.

CLI quick start
---------------

Start with a dry-run to inspect the resolved configuration without loading
weights:

.. code-block:: bash

   python examples/finetune_glm5_1_gspo.py --dry-run
   python examples/finetune_glm5_1_gspo.py --task customer_service --dry-run
   python examples/finetune_glm5_1_gspo.py --starter-profile memory --dry-run
   python examples/finetune_glm5_1_gspo.py --list-profiles

Then launch the starter run:

.. code-block:: bash

   python examples/finetune_glm5_1_gspo.py --task customer_service --no-dry-run

Starter profiles
----------------

The GLM 5.1 starter ships with three built-in profiles:

- ``balanced``: first-run QLoRA defaults, 4-bit quantization, moderate context.
- ``memory``: lower-memory variant with smaller groups and shorter context.
- ``quality``: larger context/rollout sizes for B200/H200 clusters.

Serving recommendations
-----------------------

Invoke ``get_glm5_1_serving_recommendations()`` to get vLLM-compatible flags:

.. code-block:: python

   from stateset_agents.training import get_glm5_1_serving_recommendations

   # bf16 (full precision)
   bf16 = get_glm5_1_serving_recommendations()
   # fp8 (single-host)
   fp8 = get_glm5_1_serving_recommendations(use_fp8=True)

The recommendation payload maps directly to the ``recommended`` block in
``serving_manifest.json``, which drives the Helm render scripts (see
:doc:`deployment`).

Helm values rendering
---------------------

After a training run writes a ``serving_manifest.json``, render a Helm values
override with ``scripts/render_glm5_1_helm_values.py``:

.. code-block:: bash

   python scripts/render_glm5_1_helm_values.py \
     --manifest ./outputs/glm5_1_gspo/serving_manifest.json \
     > deployment/helm/stateset-agents/values-glm5-1-finetuned.yaml

   python scripts/render_glm5_1_helm_values.py \
     --manifest ./outputs/glm5_1_gspo/serving_manifest.json \
     --gcs-uri gs://YOUR_BUCKET/glm5-1/runs/YOUR_RUN_ID/merged

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training import (
       GLM5_1_BASE_MODEL,
       GLM5_1_FP8_MODEL,
       Glm51Config,
       get_glm5_1_config,
       get_glm5_1_serving_recommendations,
       build_serving_manifest,
       export_merged_model_for_serving,
   )

   config = get_glm5_1_config(task="customer_service", starter_profile="balanced")
   serving = get_glm5_1_serving_recommendations(use_fp8=False)

Related repo files
------------------

- ``stateset_agents/training/glm5_1_starter.py``
- ``examples/finetune_glm5_1_gspo.py``
- ``scripts/render_glm5_1_helm_values.py``
- ``deployment/helm/stateset-agents/values-glm5-1*.yaml``
- ``docs/GLM5_1_HOSTING_PLAN.md``
