Qwen3.8 27B Starter Path
========================

Use this starter when you want the fastest path to a first GSPO post-training run for ``Qwen/Qwen3.8-27B``.
The recommended checkpoint for post-training is ``Qwen/Qwen3.8-27B``.

.. note::

   Qwen3.8 27B is Alibaba's open model released 2026-08-05 under Apache-2.0: a
   27.8B-parameter multimodal LM (``model_type: qwen3_5``, architecture
   ``Qwen3_5ForConditionalGeneration``) pairing a vision tower with a 64-layer
   text stack (hidden 5120, 24 heads / 4 KV heads, 248320-token vocabulary,
   262144 max positions = 256K context). The text stack uses *hybrid*
   attention: a minority of layers use standard ``self_attn``
   (``q_proj``/``k_proj``/``v_proj``/``o_proj``) while most use Mamba-style
   ``linear_attn`` (``in_proj_qkv``/``in_proj_a``/``in_proj_b``/``in_proj_z``/
   ``out_proj``); every layer has an MLP. The starter's LoRA targets cover all
   three groups, since listing only the llama-style names would silently adapt
   just the minority attention layers. The vision tower (``model.visual.*``) is
   excluded because text-only SFT sends it no gradient. The custom architecture
   requires ``trust_remote_code=True``.

.. warning::

   This is a ~56GB BF16 checkpoint. Budget roughly **160GB of disk** for the
   download plus adapters, and either a single **80GB card** (H100/A100 80GB,
   with the ``memory`` profile) or ``--gpu-count 2``. The
   ``Qwen/Qwen3.8-27B-FP8`` variant is inference-oriented; the starter emits a
   validation warning if you point it at FP8 for post-training.

CLI quick start
---------------

.. code-block:: bash

   stateset-agents qwen3-8-27b --json-output
   stateset-agents qwen3-8-27b --starter-profile memory --json-output
   stateset-agents qwen3-8-27b --list-profiles --json-output
   stateset-agents qwen3-8-27b --write-config ./qwen3_8_27b.json
   stateset-agents init --preset qwen3.8-27b --path ./qwen3_8_27b.json --format json

Run the starter
---------------

.. code-block:: bash

   stateset-agents qwen3-8-27b --no-dry-run --task customer_service
   stateset-agents qwen3-8-27b --config ./qwen3_8_27b.json --no-dry-run

Low-memory profile
------------------

.. code-block:: bash

   stateset-agents qwen3-8-27b --starter-profile memory --json-output
   python examples/finetune_gspo.py --model qwen3.8-27b --starter-profile memory --dry-run

Starter profiles
----------------

The built-in profiles are:

- ``balanced``: default Qwen3.8 27B first run with QLoRA-friendly defaults.
- ``memory``: smaller rollout groups and shorter context for tighter GPUs.
- ``quality``: larger context and rollout sizes when you have more headroom.

Example script
--------------

.. code-block:: bash

   python examples/finetune_gspo.py --model qwen3.8-27b --dry-run
   python examples/finetune_gspo.py --model qwen3.8-27b --task sales --list-profiles

Programmatic surface
--------------------

.. code-block:: python

   from stateset_agents.training.qwen3_8_starter import (
       QWEN38_27B_BASE_MODEL,
       create_qwen3_8_preview,
       describe_qwen3_8_starter_profiles,
       get_qwen3_8_config,
       load_qwen3_8_config_file,
       write_qwen3_8_config_file,
   )

   config = get_qwen3_8_config(model_name=QWEN38_27B_BASE_MODEL)
   write_qwen3_8_config_file(config, "./qwen3_8_27b.json")
   loaded = load_qwen3_8_config_file("./qwen3_8_27b.json")
   preview = create_qwen3_8_preview(loaded)
   profile_catalog = describe_qwen3_8_starter_profiles(task="sales")

Starter defaults
----------------

- output directory ``./outputs/qwen3_8_27b_gspo``
- LoRA enabled by default, targeting
  ``q_proj``/``k_proj``/``v_proj``/``o_proj`` (standard attention),
  ``in_proj_qkv``/``out_proj`` (linear attention), and
  ``gate_proj``/``up_proj``/``down_proj`` (MLP)
- 4-bit quantization enabled for the default starter profile
- task presets: ``customer_service``, ``technical_support``, ``sales``, ``conversational``

Related repo files
------------------

- ``stateset_agents/training/qwen3_8_starter.py``
- ``examples/finetune_qwen3_8_27b_gspo.py``
