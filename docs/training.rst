Training
========

Trainer implementations and training utilities live under `stateset_agents/training/`.

Start with:

.. code-block:: bash

   python examples/quick_start.py

For Kimi-K2.5:

.. code-block:: bash

   python examples/finetune_kimi_k25_gspo.py --help

For a Qwen quick start in the built docs, see :doc:`qwen3_5_starter`. The repository walkthrough remains in ``docs/QWEN3_FINETUNING_GUIDE.md``.

Offline preference optimization (DPO, IPO, SimPO, KTO, ORPO) and Bradley-Terry
reward models live in ``stateset_agents.training.preference`` and the
``stateset-agents preference`` command; see ``docs/PREFERENCE_TRAINING.md``.
Policy objectives for on-policy RL are described in ``docs/OBJECTIVES.md``.
