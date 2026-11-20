Inkling 2 Mini Starter Path
===========================

Use this starter when you want the fastest path to a first GSPO post-training run for ``thinkingmachines/Inkling-2-Mini``.
The recommended checkpoint for post-training is ``thinkingmachines/Inkling-2-Mini``.

.. note::

   Inkling 2 Mini is a multimodal looped Mixture-of-Experts model
   (``model_type: inkling2_loop``): 26.4B parameters, 24 decoder layers with
   layers 6-17 run three times, 128 routed experts with 8 active per token,
   and up to a 1M-token context window, published on HuggingFace under
   Apache-2.0. This starter covers the text modality only.

.. warning::

   This starter is a configuration integration. No Inkling 2 Mini training
   run has been verified with it.

   - The only released checkpoint stores the expert weights in NVFP4 (NVIDIA
     ModelOpt) and there is no BF16 repository. The starter therefore keeps
     the experts frozen, adapts the attention projections only, and leaves
     bitsandbytes 4-bit/8-bit loading off.
   - The model needs a Transformers runtime that ships
     ``Inkling2LoopForConditionalGeneration``. The model card does not state
     a minimum version.
   - The attention does not use llama-style ``q_proj``/``k_proj``/``v_proj``
     modules. The LoRA targets use the checkpoint's projection names
     (``wq_du``/``wk_dv``/``wv_dv``/``wo_ud``) from the safetensors weight
     map. Whether PEFT can wrap them is unverified until the model loads.
   - GGUF repositories are inference-only.

   For hosted training of the larger sibling, see the Tinker path for
   ``thinkingmachines/Inkling-2-Small`` in ``docs/MANAGED_TRAINING_PROVIDERS.md``.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents inkling-2-mini --json-output
   stateset-agents inkling-2-mini --starter-profile memory --json-output
   stateset-agents inkling-2-mini --list-profiles --json-output
   stateset-agents inkling-2-mini --write-config ./inkling2_mini.json
   stateset-agents init --preset inkling-2-mini --path ./inkling2_mini.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents inkling-2-mini --no-dry-run --task customer_service
   stateset-agents inkling-2-mini --config ./inkling2_mini.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents inkling-2-mini --starter-profile memory --json-output
   python examples/finetune_gspo.py --model inkling-2-mini --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default inkling-2-mini first run with attention-only LoRA defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_gspo.py --model inkling-2-mini --dry-run
   python examples/finetune_gspo.py --model inkling-2-mini --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.inkling2_mini_starter import (
       INKLING2_MINI_BASE_MODEL,
       create_inkling2_mini_preview,
       describe_inkling2_mini_starter_profiles,
       get_inkling2_mini_config,
       load_inkling2_mini_config_file,
       write_inkling2_mini_config_file,
   )

   config = get_inkling2_mini_config(model_name=INKLING2_MINI_BASE_MODEL)
   write_inkling2_mini_config_file(config, "./inkling2_mini.json")
   loaded = load_inkling2_mini_config_file("./inkling2_mini.json")
   preview = create_inkling2_mini_preview(loaded)
   profile_catalog = describe_inkling2_mini_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/inkling2_mini_gspo``
- LoRA enabled by default (attention projections only, rank 32)
- bitsandbytes quantization off (the checkpoint is already NVFP4-quantized)
- sampling at temperature 1.0 and top-p 1.0, matching the published generation settings
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/inkling2_mini_starter.py``
- ``examples/finetune_inkling2_mini_gspo.py``
