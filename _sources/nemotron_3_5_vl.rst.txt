Nemotron 3.5 Super VL and Bolt
=============================

Image-conditioned BF16 LoRA fine-tuning is available through
``stateset-agents nemotron-3-5-super-vl`` and
``stateset-agents nemotron-3-5-bolt``. Both commands default to a configuration
preview. Use ``--write-config`` to save settings, ``--check-data`` to validate
local image/chat JSONL, and ``--no-dry-run`` to train in a CUDA environment.

The image path is separate from the Lightning GSPO starter. See the
`complete Super VL and Bolt guide <https://github.com/stateset/stateset-agents/blob/main/docs/NEMOTRON_3_5_VL.md>`_
for installation, pinned checkpoints, data format, distributed launch, adapter
reload, and the limits of current verification.
