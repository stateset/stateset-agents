Qwen 3.5 Starter Path
=====================

Use this starter when you want the fastest path to a first GSPO post-training run for ``Qwen/Qwen3.5-0.8B``.
The recommended checkpoint for post-training is ``Qwen/Qwen3.5-0.8B-Base``.

CLI quick start
---------------

Start with a dry-run so you can inspect the resolved configuration without loading a model:

.. code-block:: bash

   stateset-agents qwen3-5-0-8b --json-output
   stateset-agents qwen3-5-0-8b --starter-profile memory --json-output
   stateset-agents qwen3-5-0-8b --list-profiles --json-output
   stateset-agents qwen3-5-0-8b --write-config ./qwen3_5_0_8b.json
   stateset-agents init --preset qwen3-5-0-8b --path ./qwen3_5_0_8b.json --format json
   stateset-agents init --preset qwen3-5-0-8b --starter-profile memory --path ./qwen3_5_0_8b_memory.json --format json

Then launch the starter run:

.. code-block:: bash

   stateset-agents qwen3-5-0-8b --no-dry-run --task customer_service
   stateset-agents qwen3-5-0-8b --config ./qwen3_5_0_8b.json --no-dry-run

If GPU memory is tight, use 4-bit quantization for the first pass:

.. code-block:: bash

   stateset-agents qwen3-5-0-8b --no-dry-run --task customer_service --use-4bit

Starter profiles
----------------

The Qwen starter supports three built-in profiles:

- ``balanced``: the default first-run settings
- ``memory``: smaller context/group sizes plus ``4-bit`` quantization for tighter GPUs
- ``quality``: larger context/group sizes when you want a heavier first run

These profiles are resolved into the saved config file, so generated configs are self-contained.

Inspect profile catalog
-----------------------

Use the built-in discovery mode when you want to compare the starter profiles before selecting one:

.. code-block:: bash

   stateset-agents qwen3-5-0-8b --task sales --list-profiles --json-output
   python examples/finetune_qwen3_5_0_8b_gspo.py --task sales --list-profiles

The catalog includes a description, validation warnings, the fully resolved config, and a short summary of effective batch size, quantization, context lengths, and group sizes.

Example script
--------------

The repo also includes an equivalent dedicated example script:

.. code-block:: bash

   python examples/finetune_qwen3_5_0_8b_gspo.py --dry-run
   python examples/finetune_qwen3_5_0_8b_gspo.py --task customer_service

Config roundtrip
------------------

The starter can write a reusable config file and load it back later:

.. code-block:: bash

   stateset-agents qwen3-5-0-8b --write-config ./qwen3_5_0_8b.json
   stateset-agents qwen3-5-0-8b --config ./qwen3_5_0_8b.json --json-output

Programmatic surface
--------------------

The packaged helper is available from ``stateset_agents.training``:

.. code-block:: python

   from stateset_agents.training import (
       QWEN35_08B_BASE_MODEL,
       create_qwen3_5_preview,
       describe_qwen3_5_starter_profiles,
       get_qwen3_5_config,
       load_qwen3_5_config_file,
       write_qwen3_5_config_file,
   )

   config = get_qwen3_5_config(model_name=QWEN35_08B_BASE_MODEL)
   write_qwen3_5_config_file(config, "./qwen3_5_0_8b.json")
   loaded = load_qwen3_5_config_file("./qwen3_5_0_8b.json")
   preview = create_qwen3_5_preview(loaded)
   profile_catalog = describe_qwen3_5_starter_profiles(task="sales")

Starter defaults
----------------

The packaged starter uses the same first-run defaults as the dedicated example path:

- ``trust_remote_code=True``
- ``attn_implementation="sdpa"``
- LoRA enabled by default
- output directory ``./outputs/qwen3_5_0_8b_gspo``
- task presets for ``customer_service``, ``technical_support``, ``sales``, and ``conversational``

Related repo files
------------------

- ``stateset_agents/training/qwen3_5_starter.py``
- ``stateset_agents/cli.py``
- ``examples/finetune_qwen3_5_0_8b_gspo.py``
- ``examples/finetune_qwen3_gspo.py``
- ``docs/QWEN3_FINETUNING_GUIDE.md``
