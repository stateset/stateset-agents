DeepSeek V4 Flash Starter Path
==============================

Use this starter when you want the fastest path to a first GSPO post-training run for ``deepseek-ai/DeepSeek-V4-Flash``.
The recommended checkpoint for post-training is ``deepseek-ai/DeepSeek-V4-Flash``.

.. note::

   DeepSeek V4 Flash is a large Mixture-of-Experts model
   (``model_type: deepseek_v4``): 43 layers, Multi-head Latent Attention
   (64 heads, 1 KV latent head), 256 routed experts with 6 active per token,
   and up to 1M max position embeddings, published on HuggingFace under MIT.
   The ``deepseek-ai/DeepSeek-V4-Flash-Base`` variant is also supported; the
   NVFP4/FP8 repos are inference-only. This starter is QLoRA-only with
   vLLM-backed generation. MLA attention does not use llama-style
   ``q_proj``/``k_proj``/``v_proj`` modules: the LoRA targets use the
   checkpoint's actual MLA projection names
   (``wq_a``/``wq_b``/``wkv``/``wo_a``/``wo_b``), verified against the
   safetensors weight map.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents deepseek-v4 --json-output
   stateset-agents deepseek-v4 --starter-profile memory --json-output
   stateset-agents deepseek-v4 --list-profiles --json-output
   stateset-agents deepseek-v4 --write-config ./deepseek_v4.json
   stateset-agents init --preset deepseek-v4 --path ./deepseek_v4.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents deepseek-v4 --no-dry-run --task customer_service
   stateset-agents deepseek-v4 --config ./deepseek_v4.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents deepseek-v4 --starter-profile memory --json-output
   python examples/finetune_gspo.py --model deepseek-v4 --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default deepseek-v4 first run with QLoRA-friendly defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_gspo.py --model deepseek-v4 --dry-run
   python examples/finetune_gspo.py --model deepseek-v4 --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.deepseek_v4_starter import (
       DEEPSEEK_V4_BASE_MODEL,
       create_deepseek_v4_preview,
       describe_deepseek_v4_starter_profiles,
       get_deepseek_v4_config,
       load_deepseek_v4_config_file,
       write_deepseek_v4_config_file,
   )

   config = get_deepseek_v4_config(model_name=DEEPSEEK_V4_BASE_MODEL)
   write_deepseek_v4_config_file(config, "./deepseek_v4.json")
   loaded = load_deepseek_v4_config_file("./deepseek_v4.json")
   preview = create_deepseek_v4_preview(loaded)
   profile_catalog = describe_deepseek_v4_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/deepseek_v4_gspo``
- QLoRA enabled by default (MLA attention projections only; 4-bit quantization)
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/deepseek_v4_starter.py``
- ``examples/finetune_deepseek_v4_gspo.py``
