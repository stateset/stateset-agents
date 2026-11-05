.. note::

   The released checkpoint is now pinned and its KDA/MLA attention targets are
   audited. Native MXFP4 loading and backward kernels still need training
   verification. Use ``stateset-agents frontier-models --model kimi-k3`` and
   :doc:`frontier_finetuning` for current readiness. Profiles do not imply
   successful full-size training.

Kimi-K3 Starter Path
====================

This starter provides configuration previews for the released, pinned
``moonshotai/Kimi-K3`` checkpoint. Exact packed-weight training is blocked
pending the loader and backward-kernel checks described above.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents kimi-k3 --json-output
   stateset-agents kimi-k3 --starter-profile memory --json-output
   stateset-agents kimi-k3 --list-profiles --json-output
   stateset-agents kimi-k3 --write-config ./kimi_k3.json
   stateset-agents init --preset kimi-k3 --path ./kimi_k3.json --format json

Inspect training readiness
--------------------------

.. code-block:: bash

   stateset-agents frontier-models --model kimi-k3

Attempting a real run with this checkpoint fails before loading model weights
with the exact unsupported-layout reason.

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents kimi-k3 --starter-profile memory --json-output
   python examples/finetune_kimi_k3_gspo.py --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default Kimi-K3 first run with configuration preview; exact MXFP4 training loader still blocked.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_kimi_k3_gspo.py --dry-run
   python examples/finetune_kimi_k3_gspo.py --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.kimi_k3_starter import (
       KIMI_K3_BASE_MODEL,
       create_kimi_k3_preview,
       describe_kimi_k3_starter_profiles,
       get_kimi_k3_config,
       load_kimi_k3_config_file,
       write_kimi_k3_config_file,
   )

   config = get_kimi_k3_config(model_name=KIMI_K3_BASE_MODEL)
   write_kimi_k3_config_file(config, "./kimi_k3.json")
   loaded = load_kimi_k3_config_file("./kimi_k3.json")
   preview = create_kimi_k3_preview(loaded)
   profile_catalog = describe_kimi_k3_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/kimi_k3_gspo``
- LoRA enabled by default
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/kimi_k3_starter.py``
- ``examples/kimi_k3_config.py``
- ``examples/finetune_kimi_k3_gspo.py``
