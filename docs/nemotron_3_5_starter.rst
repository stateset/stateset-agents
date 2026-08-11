Nemotron 3.5 Starter Path
=========================

Use this starter when you want the fastest path to a first GSPO post-training run for ``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16``.
The recommended checkpoint for post-training is ``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16``.

.. note::

   Nemotron 3.5 Lightning is NVIDIA's open model released August 2026: a hybrid
   Mamba-2 + attention + MoE causal LM (``model_type: nemotron_h``) with 30B total
   and ~3B active parameters (A3B), 52 layers, and 256K practical context (262144
   max positions; 1M claimed), published on HuggingFace under OpenMDW-1.1. The
   presets in this starter target QLoRA post-training of the BF16 checkpoint (the
   NVFP4 variant is inference-only); the custom architecture requires
   ``trust_remote_code=True``, and LoRA targets cover attention plus Mamba-2
   in/out projections.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents nemotron-3-5 --json-output
   stateset-agents nemotron-3-5 --starter-profile memory --json-output
   stateset-agents nemotron-3-5 --list-profiles --json-output
   stateset-agents nemotron-3-5 --write-config ./nemotron_3_5.json
   stateset-agents init --preset nemotron-3-5 --path ./nemotron_3_5.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents nemotron-3-5 --no-dry-run --task customer_service
   stateset-agents nemotron-3-5 --config ./nemotron_3_5.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents nemotron-3-5 --starter-profile memory --json-output
   python examples/finetune_gspo.py --model nemotron-3-5 --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default Nemotron 3.5 first run with QLoRA-friendly defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_gspo.py --model nemotron-3-5 --dry-run
   python examples/finetune_gspo.py --model nemotron-3-5 --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.nemotron_3_5_starter import (
       NEMOTRON_3_5_BASE_MODEL,
       create_nemotron_3_5_preview,
       describe_nemotron_3_5_starter_profiles,
       get_nemotron_3_5_config,
       load_nemotron_3_5_config_file,
       write_nemotron_3_5_config_file,
   )

   config = get_nemotron_3_5_config(model_name=NEMOTRON_3_5_BASE_MODEL)
   write_nemotron_3_5_config_file(config, "./nemotron_3_5.json")
   loaded = load_nemotron_3_5_config_file("./nemotron_3_5.json")
   preview = create_nemotron_3_5_preview(loaded)
   profile_catalog = describe_nemotron_3_5_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/nemotron_3_5_gspo``
- LoRA enabled by default
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/nemotron_3_5_starter.py``
- ``examples/finetune_nemotron_3_5_gspo.py``
