3D Weight Studio & MCP Control Plane
====================================

StateSet **3D Weight Studio** is an interactive WebGL2 weight/circuit inspector,
live RL training cockpit, and autonomous self-healing control plane exposed
through both a browser UI and **39 annotated MCP tools** (stdio + Streamable
HTTP) plus the ``@stateset/agents`` TypeScript SDK.

* **Full Guide (Markdown):** `docs/WEIGHT_STUDIO.md <https://github.com/stateset/stateset-agents/blob/master/docs/WEIGHT_STUDIO.md>`_
* **Getting Started (Markdown):** `docs/GETTING_STARTED_STUDIO.md <https://github.com/stateset/stateset-agents/blob/master/docs/GETTING_STARTED_STUDIO.md>`_

Key Capabilities
----------------

* **Interactive 3D Inspection & Surgery:** WebGL2 weight/circuit/attention/residual
  views, cross-checkpoint ``Split A|B`` diff, zero-download Hugging Face
  HTTP-Range ``.safetensors`` streaming, NF4/AWQ/FP8/SmoothQuant 3D quantization
  lab, SAE monosemantic feature clamping, and standalone HTML PR reports.
* **Live RL Cockpit & Bi-Directional Trainer Control:** SQLite WAL-persisted RL
  run ledger, side-by-side multi-run sparkline overlay (``ΔR``), Verifiable
  Trajectory Credit Inspector, and cooperative 4-trainer Pause / Resume /
  In-Place Weight Rollback via ``StudioControlMixin``.
* **Autonomous Production Autopilot:** Transcript-to-Surgery Auto-Heal
  (``stateset-agents heal``), 10-probe Adversarial Red-Team Immunization
  (``stateset-agents redteam``), Pareto Tournament & TIES/DARE/SLERP Merge
  (``stateset-agents tournament``), live vLLM LoRA hot-swap A/B
  (``stateset-agents serve-ab``), OTLP Trace Collector & LLM-Judge Calibration
  (``stateset-agents traces``), and Paired Shadow Canary Gate
  (``stateset-agents canary`` / ``autopilot``).

Quick Commands
--------------

.. code-block:: bash

   # Launch the interactive 3D Weight Studio & Live RL Cockpit
   stateset-agents studio --port 8765

   # Run the zero-GPU autonomous Auto-Heal, Red-Team & Canary autopilot demo
   python examples/autonomous_surgery_autopilot.py
