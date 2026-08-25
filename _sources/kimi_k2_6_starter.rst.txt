Kimi-K2.6 Starter Path
======================

Use this starter when you want the fastest path to a first GSPO post-training run for ``moonshotai/Kimi-K2.6``.
The recommended checkpoint for post-training is ``moonshotai/Kimi-K2.6``.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents kimi-k2-6 --json-output
   stateset-agents kimi-k2-6 --starter-profile memory --json-output
   stateset-agents kimi-k2-6 --list-profiles --json-output
   stateset-agents kimi-k2-6 --write-config ./kimi_k2_6.json
   stateset-agents init --preset kimi-k2-6 --path ./kimi_k2_6.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents kimi-k2-6 --no-dry-run --task customer_service
   stateset-agents kimi-k2-6 --config ./kimi_k2_6.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents kimi-k2-6 --starter-profile memory --json-output
   python examples/finetune_kimi_k2_6_gspo.py --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default Kimi-K2.6 first run with QLoRA-friendly defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_kimi_k2_6_gspo.py --dry-run
   python examples/finetune_kimi_k2_6_gspo.py --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.kimi_k2_6_starter import (
       KIMI_K26_BASE_MODEL,
       create_kimi_k2_6_preview,
       describe_kimi_k2_6_starter_profiles,
       get_kimi_k2_6_config,
       load_kimi_k2_6_config_file,
       write_kimi_k2_6_config_file,
   )

   config = get_kimi_k2_6_config(model_name=KIMI_K26_BASE_MODEL)
   write_kimi_k2_6_config_file(config, "./kimi_k2_6.json")
   loaded = load_kimi_k2_6_config_file("./kimi_k2_6.json")
   preview = create_kimi_k2_6_preview(loaded)
   profile_catalog = describe_kimi_k2_6_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/kimi_k2_6_gspo``
- LoRA enabled by default
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/kimi_k2_6_starter.py``
- ``examples/kimi_k2_6_config.py``
- ``examples/finetune_kimi_k2_6_gspo.py``
