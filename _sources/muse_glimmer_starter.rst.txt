Muse Glimmer Starter Path
=========================

Use this starter when you want the fastest path to a first GSPO post-training run for ``meta-models/Muse-Glimmer-30B``.
The recommended checkpoint for post-training is ``meta-models/Muse-Glimmer-30B``.

.. note::

   Muse Glimmer is Meta's open agentic model released August 2026: a ~30B-parameter
   dense causal transformer (52 layers, GQA 16:1, 131K+ context) with a dedicated
   perception encoder, published on HuggingFace under Apache-2.0. The presets in
   this starter target QLoRA post-training of the text stack; full-precision BF16
   inference needs ~64GB VRAM, so 4-bit quantization is on by default.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents muse-glimmer --json-output
   stateset-agents muse-glimmer --starter-profile memory --json-output
   stateset-agents muse-glimmer --list-profiles --json-output
   stateset-agents muse-glimmer --write-config ./muse_glimmer.json
   stateset-agents init --preset muse-glimmer --path ./muse_glimmer.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents muse-glimmer --no-dry-run --task customer_service
   stateset-agents muse-glimmer --config ./muse_glimmer.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents muse-glimmer --starter-profile memory --json-output
   python examples/finetune_gspo.py --model muse-glimmer --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default Muse Glimmer first run with QLoRA-friendly defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_gspo.py --model muse-glimmer --dry-run
   python examples/finetune_gspo.py --model muse-glimmer --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.muse_glimmer_starter import (
       MUSE_GLIMMER_BASE_MODEL,
       create_muse_glimmer_preview,
       describe_muse_glimmer_starter_profiles,
       get_muse_glimmer_config,
       load_muse_glimmer_config_file,
       write_muse_glimmer_config_file,
   )

   config = get_muse_glimmer_config(model_name=MUSE_GLIMMER_BASE_MODEL)
   write_muse_glimmer_config_file(config, "./muse_glimmer.json")
   loaded = load_muse_glimmer_config_file("./muse_glimmer.json")
   preview = create_muse_glimmer_preview(loaded)
   profile_catalog = describe_muse_glimmer_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/muse_glimmer_gspo``
- LoRA enabled by default
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/muse_glimmer_starter.py``
- ``examples/finetune_muse_glimmer_gspo.py``
