Qwen3 Coder Starter Path
========================

Use this starter when you want the fastest path to a first GSPO post-training run for ``Qwen/Qwen3-Coder-30B-A3B-Instruct``.
The recommended checkpoint for post-training is ``Qwen/Qwen3-Coder-30B-A3B-Instruct``.

.. note::

   Qwen3 Coder 30B is Alibaba's open coding model: a Mixture-of-Experts causal
   LM (``model_type: qwen3_moe``) with 30B total and ~3B active parameters (128
   experts, 8 active per token), 48 layers, 32 attention heads with 4 KV heads,
   and 256K context (262144 max positions), published on HuggingFace under
   Apache-2.0. The presets in this starter target QLoRA post-training of the
   BF16 checkpoint (the FP8 variant is inference-oriented). LoRA targets are
   attention-only (``q_proj``/``k_proj``/``v_proj``/``o_proj``): with 128
   experts per layer, the MoE expert MLPs are impractical LoRA targets.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents qwen3-coder --json-output
   stateset-agents qwen3-coder --starter-profile memory --json-output
   stateset-agents qwen3-coder --list-profiles --json-output
   stateset-agents qwen3-coder --write-config ./qwen3_coder.json
   stateset-agents init --preset qwen3-coder --path ./qwen3_coder.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents qwen3-coder --no-dry-run --task customer_service
   stateset-agents qwen3-coder --config ./qwen3_coder.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents qwen3-coder --starter-profile memory --json-output
   python examples/finetune_gspo.py --model qwen3-coder --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default Qwen3 Coder first run with QLoRA-friendly defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_gspo.py --model qwen3-coder --dry-run
   python examples/finetune_gspo.py --model qwen3-coder --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.qwen3_coder_starter import (
       QWEN3_CODER_BASE_MODEL,
       create_qwen3_coder_preview,
       describe_qwen3_coder_starter_profiles,
       get_qwen3_coder_config,
       load_qwen3_coder_config_file,
       write_qwen3_coder_config_file,
   )

   config = get_qwen3_coder_config(model_name=QWEN3_CODER_BASE_MODEL)
   write_qwen3_coder_config_file(config, "./qwen3_coder.json")
   loaded = load_qwen3_coder_config_file("./qwen3_coder.json")
   preview = create_qwen3_coder_preview(loaded)
   profile_catalog = describe_qwen3_coder_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/qwen3_coder_gspo``
- LoRA enabled by default (attention projections only)
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/qwen3_coder_starter.py``
- ``examples/finetune_qwen3_coder_gspo.py``
