gpt-oss Starter Path
====================

Use this starter when you want the fastest path to a first GSPO post-training run for ``openai/gpt-oss-20b``.
The recommended checkpoint for post-training is ``openai/gpt-oss-20b``.

.. note::

   gpt-oss 20B is OpenAI's open-weight reasoning model: a Mixture-of-Experts
   causal LM (``model_type: gpt_oss``) with 32 experts (4 active per token),
   24 layers, 64 attention heads with 8 KV heads, and 131072-token context,
   published on HuggingFace under Apache-2.0. The family supports adjustable
   reasoning effort and the harmony response format. The larger
   ``openai/gpt-oss-120b`` variant is also listed but needs multi-GPU
   hardware. LoRA targets are attention-only
   (``q_proj``/``k_proj``/``v_proj``/``o_proj``, verified against the
   checkpoint's weight map).

CLI quick start
---------------

.. code-block:: bash

   stateset-agents gpt-oss --json-output
   stateset-agents gpt-oss --starter-profile memory --json-output
   stateset-agents gpt-oss --list-profiles --json-output
   stateset-agents gpt-oss --write-config ./gpt_oss.json
   stateset-agents init --preset gpt-oss --path ./gpt_oss.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents gpt-oss --no-dry-run --task customer_service
   stateset-agents gpt-oss --config ./gpt_oss.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents gpt-oss --starter-profile memory --json-output
   python examples/finetune_gspo.py --model gpt-oss --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default gpt-oss first run with QLoRA-friendly defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_gspo.py --model gpt-oss --dry-run
   python examples/finetune_gspo.py --model gpt-oss --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.gpt_oss_starter import (
       GPT_OSS_BASE_MODEL,
       create_gpt_oss_preview,
       describe_gpt_oss_starter_profiles,
       get_gpt_oss_config,
       load_gpt_oss_config_file,
       write_gpt_oss_config_file,
   )

   config = get_gpt_oss_config(model_name=GPT_OSS_BASE_MODEL)
   write_gpt_oss_config_file(config, "./gpt_oss.json")
   loaded = load_gpt_oss_config_file("./gpt_oss.json")
   preview = create_gpt_oss_preview(loaded)
   profile_catalog = describe_gpt_oss_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/gpt_oss_gspo``
- LoRA enabled by default (attention projections only)
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/gpt_oss_starter.py``
- ``examples/finetune_gpt_oss_gspo.py``
