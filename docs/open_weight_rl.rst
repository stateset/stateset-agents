Open-Weight RL Suite
====================

StateSet Agents ships a unified Open-Weight RL Leadership Suite across all four
flagship trainers (``GSPOTrainer``, ``DAPOTrainer``, ``GEPOTrainer``, and
``VAPOTrainer``), paired with six concurrent deterministic MCP sandboxes and
bi-directional 3D Weight Studio controls.

* **Full Guide (Markdown):** `docs/OPEN_WEIGHT_RL.md <https://github.com/stateset/stateset-agents/blob/master/docs/OPEN_WEIGHT_RL.md>`_
* **Verified 3-Seed Ablation Artifact:** ``benchmark_results/open_weight_rl/ablation_v1_rc.json``

Key Capabilities
----------------

* **Chunked & Triton-Fused LM-Head Policy Loss:** Projects ``[B, T, H]`` hidden
  states through ``lm_head`` in token chunks (auto-dispatched single-pass Triton
  online-logsumexp kernel on CUDA with bit-exact PyTorch fallback), saving
  **99.61% peak logit VRAM** on 16k Qwen3 sequences (``37.123 GB -> 0.145 GB``)
  with ``0.0`` forward/backward log-prob delta.
* **MoE & FP8 Precision Bridge:** Native ``moe_router_aux_loss``, routing replay,
  opt-in Truncated Importance Sampling (``tis_clip_max``), and double-sided
  ``icepop_bounds`` outlier masking.
* **Six Concurrent Verifiable Enterprise Sandboxes:** ``MultiDomainSandboxPool``
  spanning ``commerce``, ``coding``, ``sql``, ``rest_api``, ``docker_code``, and
  ``web_agent`` (``BrowserDOMSandbox``).
* **Closed-Loop Adaptive RL:** ``DisaggregatedRLRuntime.run_adaptive_sandbox_loop``
  unites ``AdaptiveDifficultyCurriculum``, Turn-Level PRM Credit Assignment,
  ``OnlineRewardHackingGuard``, Step-0 PyTorch CAA surgical warm-start, vLLM
  Level-1 sleep/wake sync, and ``StudioTelemetryCallback``.

Quick Commands
--------------

.. code-block:: bash

   # Run the reproducible chunked RL & 6-domain sandbox shootout
   stateset-agents rl --run-shootout --json

   # Generate a reproducible local, Modal, RunPod, Kubernetes, River, or Colab Ultra blueprint
   stateset-agents rl --model Qwen/Qwen3-8B --algorithm gspo --cloud colab --gpu-type A100-80GB --json
   stateset-agents rl --model Qwen/Qwen3.6-35B-A3B-FP8 --algorithm cispo --cloud river --json

   # Capture SHA-256 GPU + Triton + River AI 0.11 GA evidence (JSON + Markdown)
   python benchmarks/capture_v1_gpu_river_evidence.py --skip-hf-model --river-mode offline

   # Run the end-to-end offline demo (Step-0 CAA + MCP sandboxes + Chunked GSPO)
   python examples/open_weight_rl_end_to_end.py
